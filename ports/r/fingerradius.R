# fingerradius.R - R port of the portable FingerRadiusAI core.
#
# Landmark detection is MediaPipe's job and stays in Python. What follows it --
# gesture geometry, radius maths, Kalman smoothing -- is arithmetic on 21
# points, and R is where a lot of gesture data ends up being analysed. This
# port lets an R session score recorded landmarks without shelling out.
#
# Correctness is pinned, not asserted: ../fixtures/conformance.json holds
# inputs and the outputs the Python implementation produces, and
# conformance_test.R checks this port reproduces them.
#
# Note on indexing: MediaPipe numbers landmarks 0-20 and R indexes from 1, so
# every constant below is the MediaPipe index plus one. Getting this wrong is
# the single most likely porting bug, which is why they are named rather than
# written inline.

LM <- list(
  WRIST = 1L, THUMB_MCP = 3L, THUMB_TIP = 5L,
  INDEX_MCP = 6L, INDEX_PIP = 7L, INDEX_TIP = 9L,
  MIDDLE_MCP = 10L, MIDDLE_PIP = 11L, MIDDLE_TIP = 13L,
  RING_PIP = 15L, RING_TIP = 17L,
  PINKY_PIP = 19L, PINKY_TIP = 21L
)

RADIUS_PAIRS <- list(
  list(name = "Thumb-Index",  a = LM$THUMB_TIP,  b = LM$INDEX_TIP),
  list(name = "Index-Middle", a = LM$INDEX_TIP,  b = LM$MIDDLE_TIP),
  list(name = "Middle-Ring",  a = LM$MIDDLE_TIP, b = LM$RING_TIP),
  list(name = "Ring-Pinky",   a = LM$RING_TIP,   b = LM$PINKY_TIP)
)

FINGER_JOINTS <- list(
  index  = c(LM$INDEX_TIP,  LM$INDEX_PIP),
  middle = c(LM$MIDDLE_TIP, LM$MIDDLE_PIP),
  ring   = c(LM$RING_TIP,   LM$RING_PIP),
  pinky  = c(LM$PINKY_TIP,  LM$PINKY_PIP)
)

#' Euclidean distance between two landmarks.
#' @param a,b Numeric vectors of length 2.
fr_distance <- function(a, b) sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)

#' Hand size: the wrist-to-middle-MCP bone.
#'
#' That bone is used because its length does not change as the fingers move,
#' which is exactly what a scale reference needs.
#' @param landmarks A 21x2 matrix.
fr_hand_scale <- function(landmarks) {
  max(1.0, fr_distance(landmarks[LM$WRIST, ], landmarks[LM$MIDDLE_MCP, ]))
}

#' Distances for the four adjacent fingertip pairs.
fr_radii <- function(landmarks) {
  out <- numeric(length(RADIUS_PAIRS))
  names(out) <- vapply(RADIUS_PAIRS, function(p) p$name, character(1))
  for (pair in RADIUS_PAIRS) {
    out[[pair$name]] <- fr_distance(landmarks[pair$a, ], landmarks[pair$b, ])
  }
  out
}

#' Is the thumb held out rather than folded across the palm?
#'
#' The thumb pivots sideways instead of curling, so the PIP comparison used
#' for the other fingers does not apply. How far the tip reaches from the
#' wrist relative to the thumb MCP does separate the two states.
fr_thumb_extended <- function(landmarks, thumb_extend_ratio = 1.45) {
  wrist <- landmarks[LM$WRIST, ]
  reach <- fr_distance(wrist, landmarks[LM$THUMB_TIP, ])
  knuckle <- max(1.0, fr_distance(wrist, landmarks[LM$THUMB_MCP, ]))
  (reach / knuckle) > thumb_extend_ratio
}

#' Which fingers are extended. Returns a named logical vector, thumb first.
fr_finger_states <- function(landmarks, extend_ratio = 1.12,
                             thumb_extend_ratio = 1.45) {
  wrist <- landmarks[LM$WRIST, ]
  states <- c(thumb = fr_thumb_extended(landmarks, thumb_extend_ratio))
  for (name in names(FINGER_JOINTS)) {
    joints <- FINGER_JOINTS[[name]]
    states[[name]] <- fr_distance(wrist, landmarks[joints[1], ]) >
      fr_distance(wrist, landmarks[joints[2], ]) * extend_ratio
  }
  states
}

#' Are the thumb and index tips touching?
fr_pinching <- function(landmarks, pinch_ratio = 0.28) {
  gap <- fr_distance(landmarks[LM$THUMB_TIP, ], landmarks[LM$INDEX_TIP, ])
  gap < fr_hand_scale(landmarks) * pinch_ratio
}

#' Classify a static hand pose from 21 landmarks.
#'
#' Every threshold is a ratio of hand size rather than a pixel count, so the
#' same pose classifies identically near and far from the camera.
#'
#' @param landmarks A 21x2 numeric matrix of pixel coordinates.
#' @return A list with gesture, confidence, fingers and extended_count.
fr_classify <- function(landmarks, pinch_ratio = 0.28, extend_ratio = 1.12,
                        thumb_extend_ratio = 1.45) {
  if (is.null(landmarks) || nrow(landmarks) < 21) {
    stop(sprintf("A hand has 21 landmarks; got %d",
                 if (is.null(landmarks)) 0 else nrow(landmarks)))
  }

  fingers <- fr_finger_states(landmarks, extend_ratio, thumb_extend_ratio)
  count <- sum(fingers)
  pinching <- fr_pinching(landmarks, pinch_ratio)
  scale <- fr_hand_scale(landmarks)

  gesture <- "Unknown"
  confidence <- 0.2

  # OK and pinch both touch thumb to index; the other fingers decide. Both
  # require an extended index, or a closed fist reads as a pinch.
  if (pinching && fingers[["index"]] && fingers[["middle"]] &&
      fingers[["ring"]] && fingers[["pinky"]]) {
    gesture <- "OK"; confidence <- 0.95
  } else if (fingers[["index"]] && fingers[["middle"]] && !fingers[["ring"]] &&
             !fingers[["pinky"]] && !fingers[["thumb"]]) {
    gesture <- "Peace"; confidence <- 0.95
  } else if (fingers[["thumb"]] && count == 1) {
    vertical <- landmarks[LM$THUMB_TIP, 2] - landmarks[LM$WRIST, 2]
    if (vertical < -scale * 0.5) {
      gesture <- "Thumbs Up"; confidence <- 0.92
    } else if (vertical > scale * 0.5) {
      gesture <- "Thumbs Down"; confidence <- 0.92
    } else {
      gesture <- "Unknown"; confidence <- 0.30
    }
  } else if (fingers[["index"]] && count == 1) {
    gesture <- "Pointing"; confidence <- 0.93
  } else if (pinching && fingers[["index"]]) {
    gesture <- "Pinch"; confidence <- 0.85
  } else if (count == 0) {
    gesture <- "Fist"; confidence <- 0.95
  } else if (count == 5) {
    gesture <- "Open Palm"; confidence <- 0.95
  }

  list(gesture = gesture, confidence = confidence,
       fingers = fingers, extended_count = as.integer(count))
}

#' Create a constant-velocity Kalman filter over a scalar signal.
#'
#' State is [position, velocity]. The 2x2 matrices are written out by hand:
#' at this size a matrix package would be heavier than the arithmetic.
#'
#' @return A list of closures: update(z), position(), velocity(), reset().
fr_kalman <- function(process_noise = 0.05, measurement_noise = 36.0, dt = 1.0) {
  if (process_noise <= 0) stop("process_noise must be positive")
  if (measurement_noise <= 0) stop("measurement_noise must be positive")

  position <- 0.0; velocity <- 0.0
  p00 <- 500.0; p01 <- 0.0; p10 <- 0.0; p11 <- 500.0
  initialised <- FALSE

  update <- function(z) {
    z <- as.numeric(z)

    if (!initialised) {
      # Seeding beats ramping: from zero the filter would take many frames to
      # reach the first real reading.
      position <<- z; velocity <<- 0.0
      p00 <<- 1.0; p01 <<- 0.0; p10 <<- 0.0; p11 <<- 1.0
      initialised <<- TRUE
      return(position)
    }

    # Predict.
    position <<- position + velocity * dt
    t2 <- dt * dt
    q00 <- process_noise * t2 * t2 / 4.0
    q01 <- process_noise * t2 * dt / 2.0
    q11 <- process_noise * t2

    np00 <- p00 + dt * (p10 + p01) + t2 * p11 + q00
    np01 <- p01 + dt * p11 + q01
    np10 <- p10 + dt * p11 + q01
    np11 <- p11 + q11

    # Correct. We observe position only, so H = [1, 0].
    s <- np00 + measurement_noise
    k0 <- np00 / s
    k1 <- np10 / s
    y <- z - position

    position <<- position + k0 * y
    velocity <<- velocity + k1 * y

    p00 <<- (1 - k0) * np00
    p01 <<- (1 - k0) * np01
    p10 <<- np10 - k1 * np00
    p11 <<- np11 - k1 * np01
    position
  }

  reset <- function() {
    position <<- 0.0; velocity <<- 0.0
    p00 <<- 500.0; p01 <<- 0.0; p10 <<- 0.0; p11 <<- 500.0
    initialised <<- FALSE
    invisible(NULL)
  }

  list(update = update, reset = reset,
       position = function() position, velocity = function() velocity)
}
