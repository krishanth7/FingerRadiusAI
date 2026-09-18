# frozen_string_literal: true

# finger_radius.rb - Ruby port of the portable FingerRadiusAI core.
#
# Landmark *detection* is MediaPipe's and stays in Python. Everything after
# it -- gesture geometry, radius maths, Kalman smoothing -- is arithmetic on
# 21 points, and there is no reason that has to be Python. This port lets a
# Ruby service consume landmarks from a socket, a queue or a CSV without
# standing up an interpreter it does not otherwise need.
#
# Correctness is not asserted, it is pinned: ports/fixtures/conformance.json
# holds inputs and the outputs the Python implementation produces, and
# conformance_test.rb checks this port reproduces them.

module FingerRadius
  # MediaPipe landmark indices.
  module Landmark
    WRIST = 0
    THUMB_MCP = 2
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP = 14
    RING_TIP = 16
    PINKY_PIP = 18
    PINKY_TIP = 20
  end

  # The four adjacent-fingertip pairs the dashboard graphs.
  RADIUS_PAIRS = [
    ['Thumb-Index',  Landmark::THUMB_TIP,  Landmark::INDEX_TIP],
    ['Index-Middle', Landmark::INDEX_TIP,  Landmark::MIDDLE_TIP],
    ['Middle-Ring',  Landmark::MIDDLE_TIP, Landmark::RING_TIP],
    ['Ring-Pinky',   Landmark::RING_TIP,   Landmark::PINKY_TIP]
  ].freeze

  FINGER_JOINTS = {
    'index'  => [Landmark::INDEX_TIP,  Landmark::INDEX_PIP],
    'middle' => [Landmark::MIDDLE_TIP, Landmark::MIDDLE_PIP],
    'ring'   => [Landmark::RING_TIP,   Landmark::RING_PIP],
    'pinky'  => [Landmark::PINKY_TIP,  Landmark::PINKY_PIP]
  }.freeze

  module_function

  # Euclidean distance between two [x, y] points.
  def distance(a, b)
    Math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))
  end

  # Distances for the four adjacent fingertip pairs.
  def radii(landmarks)
    RADIUS_PAIRS.each_with_object({}) do |(name, a, b), out|
      out[name] = distance(landmarks[a], landmarks[b])
    end
  end

  # Recognises static hand poses from 21 landmarks.
  #
  # Every threshold is a ratio of hand size, never a pixel count, so the same
  # pose classifies identically near and far from the camera.
  class GestureRecognizer
    FIST = 'Fist'
    OPEN_PALM = 'Open Palm'
    PEACE = 'Peace'
    THUMBS_UP = 'Thumbs Up'
    THUMBS_DOWN = 'Thumbs Down'
    OK = 'OK'
    POINTING = 'Pointing'
    PINCH = 'Pinch'
    UNKNOWN = 'Unknown'

    def initialize(pinch_ratio: 0.28, extend_ratio: 1.12, thumb_extend_ratio: 1.45)
      @pinch_ratio = pinch_ratio
      @extend_ratio = extend_ratio
      @thumb_extend_ratio = thumb_extend_ratio
    end

    # Wrist to middle-MCP: a bone whose length does not change as the fingers
    # move, which is what a scale reference needs.
    def hand_scale(landmarks)
      [1.0, FingerRadius.distance(landmarks[Landmark::WRIST],
                                  landmarks[Landmark::MIDDLE_MCP])].max
    end

    # Which fingers are extended, thumb first.
    def finger_states(landmarks)
      wrist = landmarks[Landmark::WRIST]
      states = { 'thumb' => thumb_extended?(landmarks) }
      FINGER_JOINTS.each do |name, (tip, pip)|
        states[name] = FingerRadius.distance(wrist, landmarks[tip]) >
                       FingerRadius.distance(wrist, landmarks[pip]) * @extend_ratio
      end
      states
    end

    # The thumb pivots sideways rather than curling, so the PIP comparison
    # used for the other fingers does not apply. What separates folded from
    # extended is how far the tip reaches from the wrist relative to the
    # thumb MCP.
    def thumb_extended?(landmarks)
      wrist = landmarks[Landmark::WRIST]
      reach = FingerRadius.distance(wrist, landmarks[Landmark::THUMB_TIP])
      knuckle = [1.0, FingerRadius.distance(wrist, landmarks[Landmark::THUMB_MCP])].max
      (reach / knuckle) > @thumb_extend_ratio
    end

    def pinching?(landmarks)
      gap = FingerRadius.distance(landmarks[Landmark::THUMB_TIP],
                                  landmarks[Landmark::INDEX_TIP])
      gap < hand_scale(landmarks) * @pinch_ratio
    end

    # Returns { gesture:, confidence:, fingers:, extended_count: }.
    def classify(landmarks)
      raise ArgumentError, "A hand has 21 landmarks; got #{landmarks.length}" if landmarks.length < 21

      fingers = finger_states(landmarks)
      thumb = fingers['thumb']
      index = fingers['index']
      middle = fingers['middle']
      ring = fingers['ring']
      pinky = fingers['pinky']
      count = fingers.values.count(true)
      pinching = pinching?(landmarks)
      scale = hand_scale(landmarks)

      gesture = UNKNOWN
      confidence = 0.2

      # OK and pinch both touch thumb to index; the other fingers decide. Both
      # need an extended index, or a closed fist reads as a pinch.
      if pinching && index && middle && ring && pinky
        gesture = OK
        confidence = 0.95
      elsif index && middle && !ring && !pinky && !thumb
        gesture = PEACE
        confidence = 0.95
      elsif thumb && count == 1
        vertical = landmarks[Landmark::THUMB_TIP][1] - landmarks[Landmark::WRIST][1]
        if vertical < -scale * 0.5
          gesture = THUMBS_UP
          confidence = 0.92
        elsif vertical > scale * 0.5
          gesture = THUMBS_DOWN
          confidence = 0.92
        else
          gesture = UNKNOWN
          confidence = 0.30
        end
      elsif index && count == 1
        gesture = POINTING
        confidence = 0.93
      elsif pinching && index
        gesture = PINCH
        confidence = 0.85
      elsif count.zero?
        gesture = FIST
        confidence = 0.95
      elsif count == 5
        gesture = OPEN_PALM
        confidence = 0.95
      end

      { gesture: gesture, confidence: confidence,
        fingers: fingers, extended_count: count }
    end
  end

  # Constant-velocity Kalman filter over a scalar signal.
  #
  # State is [position, velocity]. The 2x2 matrices are written out by hand
  # rather than pulled from a linear-algebra gem: at this size the dependency
  # would be larger than the maths.
  class KalmanFilter1D
    attr_reader :position, :velocity

    def initialize(process_noise: 0.05, measurement_noise: 36.0, dt: 1.0)
      raise ArgumentError, 'process_noise must be positive' unless process_noise.positive?
      raise ArgumentError, 'measurement_noise must be positive' unless measurement_noise.positive?

      @dt = dt.to_f
      @q = process_noise.to_f
      @r = measurement_noise.to_f
      @position = 0.0
      @velocity = 0.0
      # Covariance, row-major [[p00, p01], [p10, p11]].
      @p = [[500.0, 0.0], [0.0, 500.0]]
      @initialised = false
    end

    def update(measurement)
      z = measurement.to_f

      unless @initialised
        # Seeding beats ramping: starting from zero would take many frames to
        # reach the first real reading.
        @position = z
        @velocity = 0.0
        @p = [[1.0, 0.0], [0.0, 1.0]]
        @initialised = true
        return @position
      end

      # Predict: x = F x, P = F P F' + Q
      @position += @velocity * @dt
      t2 = @dt * @dt
      q00 = @q * t2 * t2 / 4.0
      q01 = @q * t2 * @dt / 2.0
      q11 = @q * t2

      p00 = @p[0][0] + (@dt * (@p[1][0] + @p[0][1])) + (t2 * @p[1][1]) + q00
      p01 = @p[0][1] + (@dt * @p[1][1]) + q01
      p10 = @p[1][0] + (@dt * @p[1][1]) + q01
      p11 = @p[1][1] + q11

      # Correct: we observe position only, so H = [1, 0].
      s = p00 + @r
      k0 = p00 / s
      k1 = p10 / s
      y = z - @position

      @position += k0 * y
      @velocity += k1 * y

      @p = [[(1 - k0) * p00, (1 - k0) * p01],
            [p10 - (k1 * p00), p11 - (k1 * p01)]]
      @position
    end

    def reset
      @position = 0.0
      @velocity = 0.0
      @p = [[500.0, 0.0], [0.0, 500.0]]
      @initialised = false
    end
  end
end
