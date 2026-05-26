import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from evdev import UInput, ecodes as e


WIDTH = 1280
HEIGHT = 720
FRAMERATE = 30

CALIBRATION_SECONDS = 3.0
PRESS_SECONDS = 0.5

# Since this version learns which ArUco id belongs to each physical button
# during calibration, it only needs detections to stay reasonably close to the
# calibrated positions. Increase this a little if the page/camera moves.
MATCH_MAX_DISTANCE_PX = 120

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])
MIN_PAGE_AREA = 10000


@dataclass(frozen=True)
class ButtonSpec:
    name: str
    output: str
    label: str


# Physical keyboard layout, top-to-bottom and left-to-right.
# This is now the source of truth instead of hardcoded ArUco ids.
LAYOUT_ROWS = [
    [
        ButtonSpec("q", "q", "Q"),
        ButtonSpec("w", "w", "W"),
        ButtonSpec("e", "e", "E"),
        ButtonSpec("r", "r", "R"),
        ButtonSpec("t", "t", "T"),
        ButtonSpec("y", "y", "Y"),
        ButtonSpec("u", "u", "U"),
        ButtonSpec("i", "i", "I"),
        ButtonSpec("o", "o", "O"),
        ButtonSpec("p", "p", "P"),
    ],
    [
        ButtonSpec("caps_lock", "caps_lock", "Caps Lock"),
        ButtonSpec("a", "a", "A"),
        ButtonSpec("s", "s", "S"),
        ButtonSpec("d", "d", "D"),
        ButtonSpec("f", "f", "F"),
        ButtonSpec("g", "g", "G"),
        ButtonSpec("h", "h", "H"),
        ButtonSpec("j", "j", "J"),
        ButtonSpec("k", "k", "K"),
        ButtonSpec("l", "l", "L"),
        ButtonSpec("enter", "enter", "Enter"),
    ],
    [
        ButtonSpec("z", "z", "Z"),
        ButtonSpec("x", "x", "X"),
        ButtonSpec("c", "c", "C"),
        ButtonSpec("v", "v", "V"),
        ButtonSpec("b", "b", "B"),
        ButtonSpec("n", "n", "N"),
        ButtonSpec("m", "m", "M"),
        ButtonSpec("backspace", "backspace", "Backspace"),
    ],
    [
        ButtonSpec("space_left", "space", "Spacebar L"),
        ButtonSpec("space_right", "space", "Spacebar R"),
    ],
]

BUTTONS = [button for row in LAYOUT_ROWS for button in row]
BUTTONS_BY_NAME = {button.name: button for button in BUTTONS}
BUTTON_PRIORITY = [button.name for button in BUTTONS]
EXPECTED_BUTTON_NAMES = set(BUTTON_PRIORITY)
TOTAL_BUTTONS = len(BUTTONS)
ROW_COUNTS = [len(row) for row in LAYOUT_ROWS]

KEY_TO_EVDEV = {
    "q": e.KEY_Q,
    "w": e.KEY_W,
    "e": e.KEY_E,
    "r": e.KEY_R,
    "t": e.KEY_T,
    "y": e.KEY_Y,
    "u": e.KEY_U,
    "i": e.KEY_I,
    "o": e.KEY_O,
    "p": e.KEY_P,
    "a": e.KEY_A,
    "s": e.KEY_S,
    "d": e.KEY_D,
    "f": e.KEY_F,
    "g": e.KEY_G,
    "h": e.KEY_H,
    "j": e.KEY_J,
    "k": e.KEY_K,
    "l": e.KEY_L,
    "z": e.KEY_Z,
    "x": e.KEY_X,
    "c": e.KEY_C,
    "v": e.KEY_V,
    "b": e.KEY_B,
    "n": e.KEY_N,
    "m": e.KEY_M,
    "enter": e.KEY_ENTER,
    "backspace": e.KEY_BACKSPACE,
    "space": e.KEY_SPACE,
}

LETTER_OUTPUTS = set("abcdefghijklmnopqrstuvwxyz")


def start_camera():
    cmd = [
        "rpicam-vid",
        "-n",
        "-t", "0",
        "--codec", "mjpeg",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FRAMERATE),
        "-o", "-",
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def mjpeg_frames(proc):
    buffer = b""

    while True:
        chunk = proc.stdout.read(4096)

        if not chunk:
            break

        buffer += chunk

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is not None:
                yield frame


def find_white_page(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    frame_h, frame_w = frame.shape[:2]
    center_x = frame_w / 2
    center_y = frame_h / 2

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < MIN_PAGE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)

        page_cx = x + w / 2
        page_cy = y + h / 2

        dist = ((page_cx - center_x) ** 2 + (page_cy - center_y) ** 2) ** 0.5
        score = area - dist * 20

        candidates.append((score, x, y, w, h, area))

    if not candidates:
        return None, mask

    candidates.sort(reverse=True)
    _, x, y, w, h, area = candidates[0]

    return (x, y, w, h, area), mask


def get_aruco_detector():
    # Keep this the same as your old working script. If the scanner says your
    # printout is a different dictionary, change this one line.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return dictionary, detector

    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_aruco_inside_page(frame, page_box, aruco_obj):
    x, y, w, h, page_area = page_box

    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    dictionary, detector_or_params = aruco_obj

    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = detector_or_params.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_or_params,
        )

    detections = []

    if ids is not None:
        ids = ids.flatten()

        for i, marker_id in enumerate(ids):
            marker_id = int(marker_id)

            pts = corners[i][0].copy()
            pts[:, 0] += x
            pts[:, 1] += y

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            detections.append({
                "aruco_id": marker_id,
                "center": (cx, cy),
                "corners": pts,
            })

    return detections


def assign_buttons_by_physical_layout(detections):
    """
    During calibration, assign marker detections to buttons by physical layout,
    not by marker id.

    It sorts detections top-to-bottom into the expected row counts, then each
    row left-to-right. This fixes the issue where the printed marker ids are
    shifted because Caps Lock / Enter / Backspace were inserted into the layout.
    """
    if len(detections) < TOTAL_BUTTONS:
        return set(), {}, {}, None

    # Use the most plausible TOTAL_BUTTONS detections. Usually there are exactly
    # 31. If OpenCV finds extras, this still tries to use the main keyboard page.
    sorted_by_y = sorted(detections, key=lambda det: det["center"][1])[:TOTAL_BUTTONS]

    rows = []
    start = 0
    for count in ROW_COUNTS:
        row = sorted_by_y[start:start + count]
        row = sorted(row, key=lambda det: det["center"][0])
        rows.append(row)
        start += count

    visible_button_names = set()
    button_centers = {}
    button_aruco_ids = {}
    row_debug = []

    for layout_row, detected_row in zip(LAYOUT_ROWS, rows):
        debug_items = []
        for button, det in zip(layout_row, detected_row):
            visible_button_names.add(button.name)
            button_centers[button.name] = det["center"]
            button_aruco_ids[button.name] = det["aruco_id"]
            debug_items.append(f"{button.label}:{det['aruco_id']}")
        row_debug.append(" | ".join(debug_items))

    return visible_button_names, button_centers, button_aruco_ids, row_debug


def assign_buttons_after_calibration(detections, calibrated_positions, calibrated_aruco_ids):
    """
    After calibration, a button is visible if a detection with the calibrated
    ArUco id appears close to that button's calibrated position.
    """
    used_detection_indices = set()
    visible_button_names = set()
    button_centers = {}

    for button_name in BUTTON_PRIORITY:
        expected_id = calibrated_aruco_ids[button_name]
        target = np.array(calibrated_positions[button_name])

        best_index = None
        best_dist = None

        for idx, det in enumerate(detections):
            if idx in used_detection_indices:
                continue
            if det["aruco_id"] != expected_id:
                continue

            center = np.array(det["center"])
            dist = float(np.linalg.norm(center - target))

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_index = idx

        if best_index is not None and best_dist <= MATCH_MAX_DISTANCE_PX:
            used_detection_indices.add(best_index)
            visible_button_names.add(button_name)
            button_centers[button_name] = detections[best_index]["center"]

    return visible_button_names, button_centers


def get_frontmost_missing_button_name(missing_button_names):
    for button_name in BUTTON_PRIORITY:
        if button_name in missing_button_names:
            return button_name

    return None


def tap_key(ui, key_code):
    ui.write(e.EV_KEY, key_code, 1)
    ui.syn()

    time.sleep(0.04)

    ui.write(e.EV_KEY, key_code, 0)
    ui.syn()


def tap_shifted_key(ui, key_code):
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1)
    ui.write(e.EV_KEY, key_code, 1)
    ui.syn()

    time.sleep(0.04)

    ui.write(e.EV_KEY, key_code, 0)
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 0)
    ui.syn()


def handle_button_press(ui, button, caps_lock_on):
    output = button.output

    if output == "caps_lock":
        caps_lock_on = not caps_lock_on
        print(f"Caps Lock toggled {'ON' if caps_lock_on else 'OFF'}.")
        return caps_lock_on

    if output not in KEY_TO_EVDEV:
        print(f"No evdev mapping for output: {output}")
        return caps_lock_on

    key_code = KEY_TO_EVDEV[output]

    if output in LETTER_OUTPUTS and caps_lock_on:
        tap_shifted_key(ui, key_code)
    else:
        tap_key(ui, key_code)

    return caps_lock_on


def draw_debug(
    frame,
    page_box,
    visible_button_names,
    button_centers,
    calibrated,
    calibration_start,
    active_button_name,
    active_missing_start,
    already_pressed,
    fps,
    caps_lock_on,
    calibrated_aruco_ids,
):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for button_name, (cx, cy) in button_centers.items():
        button = BUTTONS_BY_NAME[button_name]
        marker_id = calibrated_aruco_ids.get(button_name, "?")
        cv2.circle(debug, (cx, cy), 18, (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"{button.label}:{marker_id}",
            (cx - 35, cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    found = len(visible_button_names)
    missing_button_names = EXPECTED_BUTTON_NAMES - visible_button_names

    if not calibrated:
        if found == TOTAL_BUTTONS and calibration_start is not None:
            remaining = max(0, CALIBRATION_SECONDS - (time.time() - calibration_start))
            status = f"calibrating layout... {remaining:.1f}s"
        else:
            status = f"show all buttons: {found}/{TOTAL_BUTTONS}"
        color = (0, 255, 255)
    else:
        status = f"ready found={found}/{TOTAL_BUTTONS} missing={len(missing_button_names)}"
        color = (0, 255, 0) if len(missing_button_names) == 0 else (0, 0, 255)

    cv2.putText(
        debug,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debug,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    caps_color = (0, 255, 255) if caps_lock_on else (180, 180, 180)
    cv2.putText(
        debug,
        f"CAPS LOCK: {'ON' if caps_lock_on else 'OFF'}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        caps_color,
        2,
        cv2.LINE_AA,
    )

    if calibrated and missing_button_names:
        missing_labels = [BUTTONS_BY_NAME[name].label for name in BUTTON_PRIORITY if name in missing_button_names]
        preview = ", ".join(missing_labels[:6])
        if len(missing_labels) > 6:
            preview += ", ..."

        cv2.putText(
            debug,
            f"missing: {preview}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if calibrated and active_button_name is not None:
        button = BUTTONS_BY_NAME[active_button_name]
        elapsed = 0.0

        if active_missing_start is not None:
            elapsed = time.time() - active_missing_start

        press_status = "pressed" if active_button_name in already_pressed else "timing"

        cv2.putText(
            debug,
            f"frontmost missing: {button.label} {elapsed:.2f}s {press_status}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def main():
    print("Starting position-calibrated ArUco keyboard input.")
    print(f"Show all {TOTAL_BUTTONS} buttons for {CALIBRATION_SECONDS:.1f} seconds to calibrate.")
    print("This version learns the ArUco id under each physical key from the layout.")
    print("So shifted marker ids on the printout should not matter anymore.")
    print(f"After calibration, cover a marker for {PRESS_SECONDS:.1f} seconds to press that key.")
    print("Caps Lock toggles uppercase output inside this script until pressed again.")
    print("Press q in the video window or Ctrl+C to quit.")

    aruco_obj = get_aruco_detector()
    proc = start_camera()

    all_evdev_codes = set(KEY_TO_EVDEV.values())
    all_evdev_codes.add(e.KEY_LEFTSHIFT)

    capabilities = {
        e.EV_KEY: sorted(all_evdev_codes)
    }
    ui = UInput(capabilities, name="aruco-keyboard")

    calibrated = False
    calibrated_positions = {}
    calibrated_aruco_ids = {}

    calibration_start = None

    active_button_name = None
    active_missing_start = None
    already_pressed = set()
    caps_lock_on = False

    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    last_status_print = 0

    try:
        for frame in mjpeg_frames(proc):
            now = time.time()

            fps_frames += 1
            if now - fps_start >= 1.0:
                fps = fps_frames / (now - fps_start)
                print(f"FPS: {fps:.1f} | CAPS LOCK: {'ON' if caps_lock_on else 'OFF'}")
                fps_start = now
                fps_frames = 0

            page_box, page_mask = find_white_page(frame)

            if page_box is None:
                calibration_start = None
                active_button_name = None
                active_missing_start = None

                if now - last_status_print >= 1.0:
                    print("No white page found.")
                    last_status_print = now

                cv2.imshow("keyboard view", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            detections = detect_aruco_inside_page(frame, page_box, aruco_obj)

            if not calibrated:
                (
                    visible_button_names,
                    button_centers,
                    learned_aruco_ids,
                    row_debug,
                ) = assign_buttons_by_physical_layout(detections)

                if visible_button_names == EXPECTED_BUTTON_NAMES:
                    if calibration_start is None:
                        calibration_start = now
                        print("All buttons visible. Starting layout calibration timer.")
                        print("Current learned layout:")
                        for row in row_debug:
                            print("  " + row)

                    if now - calibration_start >= CALIBRATION_SECONDS:
                        calibrated = True
                        calibrated_positions = button_centers.copy()
                        calibrated_aruco_ids = learned_aruco_ids.copy()

                        print("Calibration complete.")
                        print("Learned button -> ArUco id mapping:")

                        for button_name in BUTTON_PRIORITY:
                            button = BUTTONS_BY_NAME[button_name]
                            print(
                                f"  {button.label:12s} "
                                f"id={calibrated_aruco_ids[button_name]:3d} "
                                f"pos={calibrated_positions[button_name]}"
                            )

                else:
                    calibration_start = None

                    if now - last_status_print >= 1.0:
                        print(f"Waiting for all buttons. Raw ArUco detections: {len(detections)}/{TOTAL_BUTTONS}.")
                        print("Make sure both spacebar markers are visible; duplicate IDs are okay.")
                        last_status_print = now

            else:
                visible_button_names, button_centers = assign_buttons_after_calibration(
                    detections,
                    calibrated_positions,
                    calibrated_aruco_ids,
                )

                missing_button_names = EXPECTED_BUTTON_NAMES - visible_button_names
                frontmost_missing_button_name = get_frontmost_missing_button_name(missing_button_names)

                if frontmost_missing_button_name is None:
                    active_button_name = None
                    active_missing_start = None

                else:
                    if frontmost_missing_button_name != active_button_name:
                        active_button_name = frontmost_missing_button_name
                        active_missing_start = now

                    button = BUTTONS_BY_NAME[active_button_name]
                    missing_time = now - active_missing_start

                    if missing_time >= PRESS_SECONDS and active_button_name not in already_pressed:
                        print(f"Pressed {button.label}. Missing for {missing_time:.2f}s.")
                        caps_lock_on = handle_button_press(ui, button, caps_lock_on)
                        already_pressed.add(active_button_name)

                visible_again = set(already_pressed) & visible_button_names

                for button_name in visible_again:
                    button = BUTTONS_BY_NAME[button_name]
                    print(f"{button.label} visible again. Can press again.")
                    already_pressed.remove(button_name)

            debug = draw_debug(
                frame=frame,
                page_box=page_box,
                visible_button_names=visible_button_names,
                button_centers=button_centers,
                calibrated=calibrated,
                calibration_start=calibration_start,
                active_button_name=active_button_name,
                active_missing_start=active_missing_start,
                already_pressed=already_pressed,
                fps=fps,
                caps_lock_on=caps_lock_on,
                calibrated_aruco_ids=calibrated_aruco_ids,
            )

            cv2.imshow("keyboard view", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping.")

    finally:
        ui.close()
        proc.terminate()
        proc.wait()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
