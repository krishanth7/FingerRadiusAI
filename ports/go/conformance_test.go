package fingerradius

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const tolerance = 1e-6

type fixture struct {
	Kalman struct {
		ProcessNoise     float64   `json:"process_noise"`
		MeasurementNoise float64   `json:"measurement_noise"`
		Input            []float64 `json:"input"`
		Expected         []float64 `json:"expected"`
	} `json:"kalman"`
	Gestures []struct {
		ID        string      `json:"id"`
		Landmarks [][]float64 `json:"landmarks"`
		Expected  struct {
			Gesture       string             `json:"gesture"`
			Fingers       map[string]bool    `json:"fingers"`
			ExtendedCount int                `json:"extended_count"`
			Radii         map[string]float64 `json:"radii"`
		} `json:"expected"`
	} `json:"gestures"`
}

func load(t *testing.T) fixture {
	t.Helper()
	path := filepath.Join("..", "fixtures", "conformance.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var f fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	return f
}

func toPoints(raw [][]float64) []Point {
	points := make([]Point, len(raw))
	for i, p := range raw {
		points[i] = Point{p[0], p[1]}
	}
	return points
}

// TestGesturesMatchPython pins this port against the Python implementation.
func TestGesturesMatchPython(t *testing.T) {
	f := load(t)
	r := NewRecognizer()

	for _, item := range f.Gestures {
		item := item
		t.Run(item.ID, func(t *testing.T) {
			landmarks := toPoints(item.Landmarks)
			got, err := r.Classify(landmarks)
			if err != nil {
				t.Fatalf("Classify: %v", err)
			}
			if got.Gesture != item.Expected.Gesture {
				t.Errorf("gesture = %q, Python says %q", got.Gesture, item.Expected.Gesture)
			}
			if got.ExtendedCount != item.Expected.ExtendedCount {
				t.Errorf("extended = %d, Python says %d", got.ExtendedCount, item.Expected.ExtendedCount)
			}
			for finger, want := range item.Expected.Fingers {
				if got.Fingers[finger] != want {
					t.Errorf("finger %s = %v, Python says %v", finger, got.Fingers[finger], want)
				}
			}
			radii := Radii(landmarks)
			for pair, want := range item.Expected.Radii {
				if math.Abs(radii[pair]-want) > tolerance {
					t.Errorf("radius %s = %.9f, Python says %.9f", pair, radii[pair], want)
				}
			}
		})
	}
}

// TestKalmanMatchesPython checks the filter reproduces Python's output to
// 1e-6 over a fixed noisy signal. Floating-point arithmetic reordered between
// languages is why this is a tolerance and not equality.
func TestKalmanMatchesPython(t *testing.T) {
	f := load(t)
	k := NewKalmanWith(f.Kalman.ProcessNoise, f.Kalman.MeasurementNoise, 1.0)

	worst := 0.0
	for i, value := range f.Kalman.Input {
		got := k.Update(value)
		delta := math.Abs(got - f.Kalman.Expected[i])
		if delta > worst {
			worst = delta
		}
		if delta > tolerance {
			t.Fatalf("step %d: got %.9f, Python says %.9f (delta %.2e)",
				i, got, f.Kalman.Expected[i], delta)
		}
	}
	t.Logf("%d steps, worst deviation from Python %.3e", len(f.Kalman.Input), worst)
}

// TestRejectsShortHand checks the guard rather than assuming it.
func TestRejectsShortHand(t *testing.T) {
	if _, err := NewRecognizer().Classify(make([]Point, 5)); err == nil {
		t.Fatal("expected an error for a 5-landmark hand")
	}
}
