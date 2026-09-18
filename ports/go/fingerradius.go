// Package fingerradius is a Go port of the portable FingerRadiusAI core.
//
// Landmark detection is MediaPipe's job and stays in Python. Everything after
// it -- gesture geometry, radius maths, Kalman smoothing -- is arithmetic on
// 21 points, which makes it a natural fit for a Go service consuming
// landmarks from a queue or a socket without embedding a Python runtime.
//
// Correctness is pinned rather than asserted: ../fixtures/conformance.json
// holds inputs and the outputs the Python implementation produces, and
// conformance_test.go checks this port reproduces them.
package fingerradius

import (
	"fmt"
	"math"
)

// MediaPipe landmark indices.
const (
	Wrist     = 0
	ThumbMCP  = 2
	ThumbTip  = 4
	IndexMCP  = 5
	IndexPIP  = 6
	IndexTip  = 8
	MiddleMCP = 9
	MiddlePIP = 10
	MiddleTip = 12
	RingPIP   = 14
	RingTip   = 16
	PinkyPIP  = 18
	PinkyTip  = 20
)

// Gesture names. Strings, matching the Python implementation exactly so a
// mixed-language deployment does not need a translation table.
const (
	Fist       = "Fist"
	OpenPalm   = "Open Palm"
	Peace      = "Peace"
	ThumbsUp   = "Thumbs Up"
	ThumbsDown = "Thumbs Down"
	OK         = "OK"
	Pointing   = "Pointing"
	Pinch      = "Pinch"
	Unknown    = "Unknown"
)

// Point is one landmark in pixels.
type Point struct{ X, Y float64 }

// RadiusPair names two fingertips whose separation is measured.
type RadiusPair struct {
	Name string
	A, B int
}

// RadiusPairs are the four adjacent-fingertip pairs the dashboard graphs.
var RadiusPairs = []RadiusPair{
	{"Thumb-Index", ThumbTip, IndexTip},
	{"Index-Middle", IndexTip, MiddleTip},
	{"Middle-Ring", MiddleTip, RingTip},
	{"Ring-Pinky", RingTip, PinkyTip},
}

var fingerJoints = []struct {
	Name     string
	Tip, PIP int
}{
	{"index", IndexTip, IndexPIP},
	{"middle", MiddleTip, MiddlePIP},
	{"ring", RingTip, RingPIP},
	{"pinky", PinkyTip, PinkyPIP},
}

// Distance returns the Euclidean distance between two landmarks.
func Distance(a, b Point) float64 {
	return math.Hypot(a.X-b.X, a.Y-b.Y)
}

// Radii returns the distance for each adjacent fingertip pair.
func Radii(landmarks []Point) map[string]float64 {
	out := make(map[string]float64, len(RadiusPairs))
	for _, pair := range RadiusPairs {
		out[pair.Name] = Distance(landmarks[pair.A], landmarks[pair.B])
	}
	return out
}

// Result is one frame's gesture verdict.
type Result struct {
	Gesture       string
	Confidence    float64
	Fingers       map[string]bool
	ExtendedCount int
}

// Recognizer classifies static hand poses from 21 landmarks.
//
// Every threshold is a ratio of hand size rather than a pixel count, so the
// same pose classifies identically near and far from the camera.
type Recognizer struct {
	PinchRatio       float64
	ExtendRatio      float64
	ThumbExtendRatio float64
}

// NewRecognizer returns a Recognizer with the tuned defaults.
func NewRecognizer() *Recognizer {
	return &Recognizer{PinchRatio: 0.28, ExtendRatio: 1.12, ThumbExtendRatio: 1.45}
}

// HandScale is the wrist-to-middle-MCP bone: a length that does not change as
// the fingers move, which is what a scale reference needs.
func (r *Recognizer) HandScale(landmarks []Point) float64 {
	return math.Max(1.0, Distance(landmarks[Wrist], landmarks[MiddleMCP]))
}

// ThumbExtended reports whether the thumb is held out rather than folded.
//
// The thumb pivots sideways instead of curling, so the PIP comparison used
// for the other fingers does not apply. How far the tip reaches from the
// wrist relative to the thumb MCP does separate the two states.
func (r *Recognizer) ThumbExtended(landmarks []Point) bool {
	wrist := landmarks[Wrist]
	reach := Distance(wrist, landmarks[ThumbTip])
	knuckle := math.Max(1.0, Distance(wrist, landmarks[ThumbMCP]))
	return reach/knuckle > r.ThumbExtendRatio
}

// FingerStates reports which fingers are extended.
func (r *Recognizer) FingerStates(landmarks []Point) map[string]bool {
	wrist := landmarks[Wrist]
	states := map[string]bool{"thumb": r.ThumbExtended(landmarks)}
	for _, joint := range fingerJoints {
		states[joint.Name] = Distance(wrist, landmarks[joint.Tip]) >
			Distance(wrist, landmarks[joint.PIP])*r.ExtendRatio
	}
	return states
}

// Pinching reports whether the thumb and index tips are touching.
func (r *Recognizer) Pinching(landmarks []Point) bool {
	gap := Distance(landmarks[ThumbTip], landmarks[IndexTip])
	return gap < r.HandScale(landmarks)*r.PinchRatio
}

// Classify names the pose in one frame.
func (r *Recognizer) Classify(landmarks []Point) (Result, error) {
	if len(landmarks) < 21 {
		return Result{}, fmt.Errorf("a hand has 21 landmarks; got %d", len(landmarks))
	}

	fingers := r.FingerStates(landmarks)
	count := 0
	for _, extended := range fingers {
		if extended {
			count++
		}
	}

	thumb, index := fingers["thumb"], fingers["index"]
	middle, ring, pinky := fingers["middle"], fingers["ring"], fingers["pinky"]
	pinching := r.Pinching(landmarks)
	scale := r.HandScale(landmarks)

	gesture, confidence := Unknown, 0.2

	switch {
	// OK and pinch both touch thumb to index; the other fingers decide. Both
	// require an extended index, or a closed fist reads as a pinch.
	case pinching && index && middle && ring && pinky:
		gesture, confidence = OK, 0.95
	case index && middle && !ring && !pinky && !thumb:
		gesture, confidence = Peace, 0.95
	case thumb && count == 1:
		vertical := landmarks[ThumbTip].Y - landmarks[Wrist].Y
		switch {
		case vertical < -scale*0.5:
			gesture, confidence = ThumbsUp, 0.92
		case vertical > scale*0.5:
			gesture, confidence = ThumbsDown, 0.92
		default:
			gesture, confidence = Unknown, 0.30
		}
	case index && count == 1:
		gesture, confidence = Pointing, 0.93
	case pinching && index:
		gesture, confidence = Pinch, 0.85
	case count == 0:
		gesture, confidence = Fist, 0.95
	case count == 5:
		gesture, confidence = OpenPalm, 0.95
	}

	return Result{gesture, confidence, fingers, count}, nil
}

// Kalman is a constant-velocity Kalman filter over a scalar signal.
//
// State is [position, velocity]. The 2x2 matrices are written out by hand:
// at this size a linear-algebra dependency would be larger than the maths.
type Kalman struct {
	dt, q, r           float64
	position, velocity float64
	p00, p01, p10, p11 float64
	initialised        bool
}

// NewKalman returns a filter with the tuned defaults, which beat the EMA they
// replace by 14.7% mean RMSE across the signals in the Python test suite.
func NewKalman() *Kalman { return NewKalmanWith(0.05, 36.0, 1.0) }

// NewKalmanWith returns a filter with explicit noise parameters.
func NewKalmanWith(processNoise, measurementNoise, dt float64) *Kalman {
	return &Kalman{
		dt: dt, q: processNoise, r: measurementNoise,
		p00: 500.0, p11: 500.0,
	}
}

// Update folds in one measurement and returns the filtered position.
func (k *Kalman) Update(measurement float64) float64 {
	if !k.initialised {
		// Seeding beats ramping: from zero the filter would take many frames
		// to reach the first real reading.
		k.position, k.velocity = measurement, 0
		k.p00, k.p01, k.p10, k.p11 = 1, 0, 0, 1
		k.initialised = true
		return k.position
	}

	// Predict.
	k.position += k.velocity * k.dt
	t2 := k.dt * k.dt
	q00 := k.q * t2 * t2 / 4.0
	q01 := k.q * t2 * k.dt / 2.0
	q11 := k.q * t2

	p00 := k.p00 + k.dt*(k.p10+k.p01) + t2*k.p11 + q00
	p01 := k.p01 + k.dt*k.p11 + q01
	p10 := k.p10 + k.dt*k.p11 + q01
	p11 := k.p11 + q11

	// Correct. We observe position only, so H = [1, 0].
	s := p00 + k.r
	k0 := p00 / s
	k1 := p10 / s
	y := measurement - k.position

	k.position += k0 * y
	k.velocity += k1 * y

	k.p00 = (1 - k0) * p00
	k.p01 = (1 - k0) * p01
	k.p10 = p10 - k1*p00
	k.p11 = p11 - k1*p01
	return k.position
}

// Position returns the current filtered estimate.
func (k *Kalman) Position() float64 { return k.position }

// Velocity returns the current estimated velocity, per timestep.
func (k *Kalman) Velocity() float64 { return k.velocity }

// Reset forgets all state; the next Update re-seeds the filter.
func (k *Kalman) Reset() {
	k.position, k.velocity = 0, 0
	k.p00, k.p01, k.p10, k.p11 = 500, 0, 0, 500
	k.initialised = false
}
