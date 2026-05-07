import time
import subprocess
import cv2
import numpy as np
from evdev import UInput, ecodes as e


FRAME_PATH = "/tmp/keyboard_frame.jpg"

WIDTH = 640
HEIGHT = 480

ORANGE_LOWER = np.array([5, 80, 80])
ORANGE_UPPER = np.array([30, 255, 255])

DOT_RADIUS = 16
VISIBLE_THRESHOLD = 35

MIN_DOT_AREA = 20
MAX_DOT_AREA = 2500

ROW_Y_TOL = 35

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


def capture_frame():
    cmd = [
        "rpicam-still",
        "-n",
        "--timeout", "100",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "-o", FRAME_PATH,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("rpicam-still failed:")
        print(result.stderr.decode(errors="ignore"))
        return None

    frame = cv2.imread(FRAME_PATH)

    if frame is None:
        print("cv2 could not read frame.")
        return None

    return frame


def orange_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def find_orange_dots(frame):
    mask = orange_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []

    for c in contours:
        area = cv2.contourArea(c)

        if MIN_DOT_AREA < area < MAX_DOT_AREA:
            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dots.append((cx, cy, area))

    return dots, mask


def group_rows(dots):
    points = [(x, y) for x, y, area in dots]
    points = sorted(points, key=lambda p: p[1])

    rows = []

    for point in points:
        x, y = point
        placed = False

        for row in rows:
            avg_y = sum(p[1] for p in row) / len(row)

            if abs(y - avg_y) < ROW_Y_TOL:
                row.append(point)
                placed = True
                break

        if not placed:
            rows.append([point])

    for row in rows:
        row.sort(key=lambda p: p[0])

    rows.sort(key=lambda row: sum(p[1] for p in row) / len(row))

    return rows


def assign_keys_from_rows(rows):
    key_points = {}

    space_rows = [row for row in rows if len(row) == 2]
    letter_rows = [row for row in rows if len(row) != 2]

    print("Rows found:", [len(row) for row in rows])

    if len(space_rows) != 1:
        print("Expected exactly one row with 2 spacebar dots.")
        return None

    if len(letter_rows) != 3:
        print("Expected exactly 3 letter rows.")
        return None

    letter_rows.sort(key=lambda row: sum(p[1] for p in row) / len(row))

    expected_counts = [10, 9, 7]

    for row, expected in zip(letter_rows, expected_counts):
        if len(row) != expected:
            print("Bad row count.")
            print("Expected:", expected)
            print("Got:", len(row))
            return None

    for row, labels in zip(letter_rows, KEY_ROWS):
        row = sorted(row, key=lambda p: p[0])

        for point, label in zip(row, labels):
            key_points[label] = point

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


def draw_debug(frame, key_points, visible):
    debug = frame.copy()

    for key, (x, y) in key_points.items():
        if visible.get(key, True):
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(debug, (x, y), DOT_RADIUS, color, 2)
        cv2.putText(
            debug,
            key,
            (x - 12, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite("debug_keyboard.jpg", debug)


def main():
    ui = UInput()

    key_points = None
    locked = False

    print("Starting.")
    print("Make sure all orange dots are visible for calibration.")

    while True:
        frame = capture_frame()

        if frame is None:
            print("Could not capture frame.")
            time.sleep(0.2)
            continue

        dots, mask = find_orange_dots(frame)

        print("Orange dots found:", len(dots))

        cv2.imwrite("debug_frame.jpg", frame)
        cv2.imwrite("debug_orange_mask.jpg", mask)

        if key_points is None:
            rows = group_rows(dots)
            key_points = assign_keys_from_rows(rows)

            if key_points is None:
                print("Calibration failed. Retrying...")
                time.sleep(0.5)
                continue

            print("Calibrated keys:")

            for key, point in key_points.items():
                print(key, point)

            print("Ready.")
            cv2.imwrite("calibrated_keyboard.jpg", frame)

        visible = {
            key: dot_visible(mask, point)
            for key, point in key_points.items()
        }

        draw_debug(frame, key_points, visible)

        all_visible = all(visible.values())

        if locked:
            if all_visible:
                locked = False
                print("Reset. Ready for next key.")

            time.sleep(0.05)
            continue

        missing = [key for key in key_points if not visible[key]]

        if missing:
            pressed = missing[0]

            if pressed in ["space_1", "space_2"]:
                pressed = "space"

            print("Pressed:", pressed)
            send_key(ui, pressed)
            locked = True

        time.sleep(0.05)


if __name__ == "__main__":
    main()