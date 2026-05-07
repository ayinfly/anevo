import time
import cv2
import numpy as np
from picamera2 import Picamera2
from evdev import UInput, ecodes as e


OUT_W, OUT_H = 900, 600

ORANGE_LOWER = np.array([5, 80, 80])
ORANGE_UPPER = np.array([30, 255, 255])

DOT_RADIUS = 16
VISIBLE_THRESHOLD = 35

KEY_ROWS = [
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm"),
]

EVDEV_KEYS = {
    "q": e.KEY_Q, "w": e.KEY_W, "e": e.KEY_E, "r": e.KEY_R, "t": e.KEY_T,
    "y": e.KEY_Y, "u": e.KEY_U, "i": e.KEY_I, "o": e.KEY_O, "p": e.KEY_P,
    "a": e.KEY_A, "s": e.KEY_S, "d": e.KEY_D, "f": e.KEY_F, "g": e.KEY_G,
    "h": e.KEY_H, "j": e.KEY_J, "k": e.KEY_K, "l": e.KEY_L,
    "z": e.KEY_Z, "x": e.KEY_X, "c": e.KEY_C, "v": e.KEY_V, "b": e.KEY_B,
    "n": e.KEY_N, "m": e.KEY_M,
    "space": e.KEY_SPACE,
}


def get_frame(picam2):
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_corner_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Assumes corner squares are dark/black.
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)

        if 0.55 < ratio < 1.8:
            candidates.append((x, y, w, h, area))

    if len(candidates) < 4:
        return None

    candidates = sorted(candidates, key=lambda b: b[4], reverse=True)[:4]

    centers = []
    for x, y, w, h, area in candidates:
        centers.append((x + w / 2, y + h / 2))

    return order_points(centers)


def warp_keyboard(frame, corners):
    dst = np.array([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (OUT_W, OUT_H))
    return warped


def orange_mask(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def find_orange_dots(warped):
    mask = orange_mask(warped)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []

    for c in contours:
        area = cv2.contourArea(c)

        if 20 < area < 2500:
            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dots.append((cx, cy, area))

    return dots, mask


def group_rows(dots, y_tol=35):
    points = [(x, y) for x, y, area in dots]
    points = sorted(points, key=lambda p: p[1])

    rows = []

    for p in points:
        x, y = p
        placed = False

        for row in rows:
            avg_y = sum(pt[1] for pt in row) / len(row)

            if abs(y - avg_y) < y_tol:
                row.append(p)
                placed = True
                break

        if not placed:
            rows.append([p])

    for row in rows:
        row.sort(key=lambda p: p[0])

    rows.sort(key=lambda row: sum(p[1] for p in row) / len(row))

    return rows


def assign_keys_from_rows(rows):
    key_points = {}

    # Space row is the row with exactly 2 dots.
    space_rows = [row for row in rows if len(row) == 2]
    letter_rows = [row for row in rows if len(row) != 2]

    if len(space_rows) != 1:
        print("Expected one space row with exactly 2 dots.")
        return None

    # Sort letter rows top-to-bottom.
    letter_rows.sort(key=lambda row: sum(p[1] for p in row) / len(row))

    expected_counts = [10, 9, 7]

    if len(letter_rows) != 3:
        print("Expected 3 letter rows, but found", len(letter_rows))
        return None

    for row, expected in zip(letter_rows, expected_counts):
        if len(row) != expected:
            print("Bad row count. Expected", expected, "but got", len(row))
            print("Rows found:", [len(r) for r in rows])
            return None

    for row, labels in zip(letter_rows, KEY_ROWS):
        row = sorted(row, key=lambda p: p[0])

        for point, label in zip(row, labels):
            key_points[label] = point

    # Two space dots both map to the same space key.
    # If either space dot is covered, it will type space.
    space_points = sorted(space_rows[0], key=lambda p: p[0])
    key_points["space_1"] = space_points[0]
    key_points["space_2"] = space_points[1]

    return key_points


def dot_visible(mask, point):
    x, y = point

    x1 = max(0, x - DOT_RADIUS)
    x2 = min(mask.shape[1], x + DOT_RADIUS)
    y1 = max(0, y - DOT_RADIUS)
    y2 = min(mask.shape[0], y + DOT_RADIUS)

    roi = mask[y1:y2, x1:x2]
    orange_pixels = cv2.countNonZero(roi)

    return orange_pixels > VISIBLE_THRESHOLD


def send_key(ui, key):
    if key.startswith("space"):
        key = "space"

    code = EVDEV_KEYS[key]

    ui.write(e.EV_KEY, code, 1)
    ui.syn()

    time.sleep(0.04)

    ui.write(e.EV_KEY, code, 0)
    ui.syn()


def draw_debug(warped, key_points, visible):
    debug = warped.copy()

    for key, (x, y) in key_points.items():
        color = (0, 255, 0) if visible.get(key, True) else (0, 0, 255)

        cv2.circle(debug, (x, y), DOT_RADIUS, color, 2)
        cv2.putText(
            debug,
            key,
            (x - 10, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite("debug_keyboard.jpg", debug)


def main():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )
    picam2.start()

    ui = UInput()

    key_points = None
    locked = False

    print("Starting.")
    print("Make sure all orange dots are visible for calibration.")

    while True:
        frame = get_frame(picam2)

        corners = find_corner_squares(frame)

        if corners is None:
            print("Could not find 4 corner squares.")
            time.sleep(0.2)
            continue

        warped = warp_keyboard(frame, corners)
        dots, mask = find_orange_dots(warped)

        if key_points is None:
            rows = group_rows(dots)
            print("Rows found:", [len(r) for r in rows])

            key_points = assign_keys_from_rows(rows)

            if key_points is None:
                print("Calibration failed. Retrying...")
                time.sleep(0.5)
                continue

            print("Calibrated keys:")
            for key, point in key_points.items():
                print(key, point)

            print("Ready.")
            cv2.imwrite("calibrated_keyboard.jpg", warped)

        visible = {
            key: dot_visible(mask, point)
            for key, point in key_points.items()
        }

        draw_debug(warped, key_points, visible)

        all_visible = all(visible.values())

        if locked:
            if all_visible:
                locked = False
                print("Reset. Ready for next key.")
            time.sleep(0.03)
            continue

        missing = [key for key in key_points if not visible[key]]

        if missing:
            pressed = missing[0]

            if pressed in ["space_1", "space_2"]:
                pressed = "space"

            print("Pressed:", pressed)
            send_key(ui, pressed)
            locked = True

        time.sleep(0.03)


if __name__ == "__main__":
    main()