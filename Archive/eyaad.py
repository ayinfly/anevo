import cv2
import numpy as np
import time
import math

# ============================================================
# Paper Keyboard MVP
# One camera at 45 degrees
# Orange dots = keyboard key centers
# Bare finger detection = skin color + motion
# ============================================================

# -----------------------------
# Camera/settings
# -----------------------------
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Keep False while testing.
# Set True later if you want actual computer keyboard input.
SEND_KEYS = False

if SEND_KEYS:
    from pynput.keyboard import Controller, Key
    keyboard = Controller()

# -----------------------------
# Press detection tuning
# -----------------------------
PRESS_DOWN_PIXELS = 10
PRESS_COOLDOWN = 0.35
MAX_KEY_DISTANCE = 70

# -----------------------------
# Blob detection tuning
# -----------------------------
MIN_DOT_AREA = 40
MAX_DOT_AREA = 1500
MIN_FINGER_AREA = 700

# -----------------------------
# Orange dot HSV range
# -----------------------------
LOWER_ORANGE = np.array([5, 80, 80])
UPPER_ORANGE = np.array([30, 255, 255])

# -----------------------------
# Keyboard layout
# -----------------------------
KEY_ROWS = [
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm"),
]

# 10 letters + 9 letters + 7 letters + 2 spacebar dots
EXPECTED_ROW_COUNTS = [10, 9, 7, 2]


# ============================================================
# Utility functions
# ============================================================

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def press_key(key_name):
    if not SEND_KEYS:
        return

    if key_name == "space":
        keyboard.press(Key.space)
        keyboard.release(Key.space)
    else:
        keyboard.press(key_name)
        keyboard.release(key_name)


# ============================================================
# Orange dot detection
# ============================================================

def detect_orange_dots(frame):
    """
    Detects orange dots printed on the keyboard.
    Returns:
        centers: list of (x, y)
        mask: orange mask
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < MIN_DOT_AREA or area > MAX_DOT_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        centers.append((cx, cy))

    return centers, mask


def cluster_rows_by_y(points, k=4):
    """
    Clusters detected dots into rows using y coordinate.
    For this MVP:
        row 0 = qwertyuiop
        row 1 = asdfghjkl
        row 2 = zxcvbnm
        row 3 = spacebar dots
    """
    if len(points) < k:
        return None

    data = np.float32([[p[1]] for p in points])

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2
    )

    _, labels, _ = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    rows = [[] for _ in range(k)]

    for pt, label in zip(points, labels.flatten()):
        rows[label].append(pt)

    rows.sort(key=lambda row: np.mean([p[1] for p in row]))

    return rows


def assign_dots_to_keys(dot_centers):
    """
    Assigns orange dots to keyboard keys.
    Expects:
        10 top-row dots
        9 middle-row dots
        7 bottom-row dots
        2 spacebar dots below all others
    """
    expected_total = sum(EXPECTED_ROW_COUNTS)

    if len(dot_centers) != expected_total:
        print(f"Expected {expected_total} orange dots, but found {len(dot_centers)}.")
        return None

    rows = cluster_rows_by_y(dot_centers, k=4)

    if rows is None:
        return None

    for i, row in enumerate(rows):
        expected = EXPECTED_ROW_COUNTS[i]
        if len(row) != expected:
            print(f"Row {i + 1} expected {expected} dots, but found {len(row)}.")
            return None

    key_positions = {}

    # Letter rows
    for row_idx in range(3):
        row_points = sorted(rows[row_idx], key=lambda p: p[0])
        labels = KEY_ROWS[row_idx]

        for label, pt in zip(labels, row_points):
            key_positions[label] = pt

    # Spacebar row: average the two bottom dots into one key center
    space_points = sorted(rows[3], key=lambda p: p[0])
    sx = int(np.mean([p[0] for p in space_points]))
    sy = int(np.mean([p[1] for p in space_points]))
    key_positions["space"] = (sx, sy)

    return key_positions


# ============================================================
# Bare finger detection
# ============================================================

def make_skin_mask(frame):
    """
    Detects skin-colored regions using YCrCb thresholding.

    This is not perfect and depends on lighting.
    If it detects too much or too little skin, tune the thresholds below.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Common simple skin threshold range.
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    return skin_mask


def make_motion_mask(frame, background_gray):
    """
    Detects moving objects compared to the saved background.
    The background should be captured when no hand is visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, background_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    return motion_mask


def detect_fingertip(frame, background_gray):
    """
    Detects bare finger by combining:
        skin color mask
        motion mask

    Returns:
        fingertip: (x, y), or None
        debug_mask
    """
    skin_mask = make_skin_mask(frame)
    motion_mask = make_motion_mask(frame, background_gray)

    # Combine skin and motion so static skin-colored objects are ignored.
    mask = cv2.bitwise_and(skin_mask, motion_mask)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_FINGER_AREA:
        return None, mask

    # Since the camera is angled, the lowest point of the hand/finger blob
    # is usually closest to the fingertip touching the paper.
    bottom_point = largest[largest[:, :, 1].argmax()][0]
    fingertip = (int(bottom_point[0]), int(bottom_point[1]))

    return fingertip, mask


# ============================================================
# Key lookup and press detection
# ============================================================

def nearest_key(fingertip, key_positions):
    if fingertip is None:
        return None, None

    best_key = None
    best_dist = float("inf")

    for key, pt in key_positions.items():
        d = distance(fingertip, pt)

        if d < best_dist:
            best_dist = d
            best_key = key

    if best_dist > MAX_KEY_DISTANCE:
        return None, best_dist

    return best_key, best_dist


class PressDetector:
    def __init__(self):
        self.last_key = None
        self.base_y = None
        self.last_press_time = 0
        self.is_down = False

    def update(self, current_key, fingertip_y):
        if current_key is None or fingertip_y is None:
            self.last_key = None
            self.base_y = None
            self.is_down = False
            return None

        now = time.time()

        if self.last_key != current_key:
            self.last_key = current_key
            self.base_y = fingertip_y
            self.is_down = False
            return None

        if self.base_y is None:
            self.base_y = fingertip_y
            return None

        dy = fingertip_y - self.base_y

        pressed = (
            dy > PRESS_DOWN_PIXELS
            and not self.is_down
            and now - self.last_press_time > PRESS_COOLDOWN
        )

        if pressed:
            self.is_down = True
            self.last_press_time = now
            self.base_y = fingertip_y
            return current_key

        # Reset when finger comes back up.
        if dy < -4:
            self.is_down = False
            self.base_y = fingertip_y

        return None


# ============================================================
# Drawing helpers
# ============================================================

def draw_key_map(frame, key_positions, current_key=None):
    for key, pt in key_positions.items():
        color = (0, 255, 0)
        radius = 8

        if key == current_key:
            color = (0, 255, 255)
            radius = 13

        cv2.circle(frame, pt, radius, color, 2)

        display_label = key
        if key == "space":
            display_label = "space"

        cv2.putText(
            frame,
            display_label,
            (pt[0] - 15, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


def draw_status(frame, lines):
    y = 30

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30


# ============================================================
# Calibration helpers
# ============================================================

def capture_background(cap):
    """
    Captures a background frame with no hand visible.
    """
    print()
    print("Background capture:")
    print("Remove your hand from the frame.")
    print("Press 'b' to capture the background.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        preview = frame.copy()

        draw_status(preview, [
            "Remove hand from frame",
            "Press 'b' to capture background",
            "Press 'q' to quit"
        ])

        cv2.imshow("background capture", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None

        if key == ord("b"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.destroyWindow("background capture")
            print("Background captured.")
            return gray


def calibrate_key_dots(cap):
    """
    Detects orange dots and assigns them to keys.
    """
    print()
    print("Dot calibration:")
    print("Make sure all orange dots are visible.")
    print("Make sure your hand is NOT in the frame.")
    print("Press 'c' to calibrate.")
    print("Press 'q' to quit.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        dots, dot_mask = detect_orange_dots(frame)

        preview = frame.copy()

        for p in dots:
            cv2.circle(preview, p, 6, (0, 255, 0), 2)

        draw_status(preview, [
            f"Detected orange dots: {len(dots)} / 28",
            "Press 'c' to calibrate",
            "Press 'q' to quit"
        ])

        cv2.imshow("dot calibration", preview)
        cv2.imshow("orange dot mask", dot_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None

        if key == ord("c"):
            key_positions = assign_dots_to_keys(dots)

            if key_positions is None:
                print("Calibration failed.")
                print("Make sure the code sees exactly:")
                print("- 10 dots in Q row")
                print("- 9 dots in A row")
                print("- 7 dots in Z row")
                print("- 2 dots below for spacebar")
            else:
                cv2.destroyWindow("dot calibration")
                cv2.destroyWindow("orange dot mask")
                print("Dot calibration successful.")
                return key_positions


# ============================================================
# Main
# ============================================================

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    key_positions = calibrate_key_dots(cap)

    if key_positions is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    background_gray = capture_background(cap)

    if background_gray is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    press_detector = PressDetector()

    print()
    print("Running paper keyboard.")
    print("Press 'r' to recapture background.")
    print("Press 'q' to quit.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        fingertip, finger_mask = detect_fingertip(frame, background_gray)

        current_key = None
        pressed_key = None

        if fingertip is not None:
            current_key, dist = nearest_key(fingertip, key_positions)

            pressed_key = press_detector.update(
                current_key,
                fingertip[1]
            )

            cv2.circle(frame, fingertip, 8, (255, 0, 0), -1)

        else:
            press_detector.update(None, None)

        draw_key_map(frame, key_positions, current_key=current_key)

        status_lines = [
            f"Current key: {current_key}",
            "Press 'r' to recapture background",
            "Press 'q' to quit"
        ]

        draw_status(frame, status_lines)

        if pressed_key is not None:
            print(f"Pressed: {pressed_key}")
            press_key(pressed_key)

            cv2.putText(
                frame,
                f"PRESSED: {pressed_key}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        cv2.imshow("paper keyboard", frame)
        cv2.imshow("finger mask", finger_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            new_background = capture_background(cap)
            if new_background is not None:
                background_gray = new_background
                print("Background recaptured.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
