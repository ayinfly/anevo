import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from evdev import UInput, ecodes as e


WIDTH = 640
HEIGHT = 360
FRAMERATE = 30

CALIBRATION_SECONDS = 3.0
PRESS_SECONDS = 0.1

# Used only when the same ArUco id appears more than once on the page.
# Example: if both spacebar markers are the same printed marker, this lets the
# code distinguish them by their calibrated locations.
MATCH_MAX_DISTANCE_PX = 90

# If both printed spacebar markers are the exact same ArUco marker, put that
# one marker id here. Run scan_aruco_ids.py and use the duplicate id it prints
# for the two spacebar markers.
SPACEBAR_ARUCO_ID = 29

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])
MIN_PAGE_AREA = 10000


@dataclass(frozen=True)
class ButtonSpec:
    # Unique internal name used by the code.
    name: str

    # ArUco marker id printed on the paper.
    aruco_id: int

    # What this button sends/toggles.
    # Letters are "q".."z". Special values are "caps_lock", "enter",
    # "backspace", and "space".
    output: str

    # Text shown in debug overlay / terminal.
    label: str


# IMPORTANT:
# These ids must match the ArUco markers on your printed keyboard.
# Existing letters stay as ids 0-25. I added:
#   26 = Caps Lock
#   27 = Enter
#   28 = Backspace
#   SPACEBAR_ARUCO_ID = both Spacebar markers
# If your printout uses different ids, only change the aruco_id numbers below.
# Since your two spacebar markers are intentionally the same, this script
# assigns the two same-id detections by their calibrated left/right positions.
BUTTONS = [
    ButtonSpec("q", 0, "q", "Q"),
    ButtonSpec("w", 1, "w", "W"),
    ButtonSpec("e", 2, "e", "E"),
    ButtonSpec("r", 3, "r", "R"),
    ButtonSpec("t", 4, "t", "T"),
    ButtonSpec("y", 5, "y", "Y"),
    ButtonSpec("u", 6, "u", "U"),
    ButtonSpec("i", 7, "i", "I"),
    ButtonSpec("o", 8, "o", "O"),
    ButtonSpec("p", 9, "p", "P"),

    ButtonSpec("caps_lock", 26, "caps_lock", "Caps Lock"),
    ButtonSpec("a", 10, "a", "A"),
    ButtonSpec("s", 11, "s", "S"),
    ButtonSpec("d", 12, "d", "D"),
    ButtonSpec("f", 13, "f", "F"),
    ButtonSpec("g", 14, "g", "G"),
    ButtonSpec("h", 15, "h", "H"),
    ButtonSpec("j", 16, "j", "J"),
    ButtonSpec("k", 17, "k", "K"),
    ButtonSpec("l", 18, "l", "L"),
    ButtonSpec("enter", 27, "enter", "Enter"),

    ButtonSpec("z", 19, "z", "Z"),
    ButtonSpec("x", 20, "x", "X"),
    ButtonSpec("c", 21, "c", "C"),
    ButtonSpec("v", 22, "v", "V"),
    ButtonSpec("b", 23, "b", "B"),
    ButtonSpec("n", 24, "n", "N"),
    ButtonSpec("m", 25, "m", "M"),
    ButtonSpec("backspace", 28, "backspace", "Backspace"),

    ButtonSpec("space_left", SPACEBAR_ARUCO_ID, "space", "Spacebar L"),
    ButtonSpec("space_right", SPACEBAR_ARUCO_ID, "space", "Spacebar R"),
]

BUTTONS_BY_NAME = {button.name: button for button in BUTTONS}
SPECS_BY_ARUCO_ID = defaultdict(list)
for button in BUTTONS:
    SPECS_BY_ARUCO_ID[button.aruco_id].append(button)

EXPECTED_BUTTON_NAMES = {button.name for button in BUTTONS}
EXPECTED_ARUCO_IDS = {button.aruco_id for button in BUTTONS}
EXPECTED_ARUCO_COUNTS = Counter(button.aruco_id for button in BUTTONS)
TOTAL_BUTTONS = len(BUTTONS)

# Press priority when a finger/hand covers multiple markers. This keeps the old
# behavior: top row first, then home row, then bottom row, then spacebar.
BUTTON_PRIORITY = [button.name for button in BUTTONS]

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

            if marker_id not in EXPECTED_ARUCO_IDS:
                continue

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


def group_detections_by_aruco_id(detections):
    detections_by_id = defaultdict(list)

    for det in detections:
        detections_by_id[det["aruco_id"]].append(det)

    for marker_id in detections_by_id:
        # Stable left-to-right order. This matters when the same printed ArUco
        # id appears twice, like the two spacebar markers.
        detections_by_id[marker_id].sort(
            key=lambda det: det["center"][0]
        )

    return detections_by_id


def assign_buttons_for_calibration(detections):
    """
    Assign visible ArUco detections to button names before calibration exists.

    For unique marker ids, this is direct. For duplicate marker ids, we sort by
    location and assign to the duplicate ButtonSpec entries in BUTTONS order.
    """
    detections_by_id = group_detections_by_aruco_id(detections)

    visible_button_names = set()
    button_centers = {}

    for marker_id, specs in SPECS_BY_ARUCO_ID.items():
        dets = detections_by_id.get(marker_id, [])

        if len(dets) < len(specs):
            continue

        specs_sorted = sorted(specs, key=lambda spec: BUTTON_PRIORITY.index(spec.name))

        for spec, det in zip(specs_sorted, dets):
            visible_button_names.add(spec.name)
            button_centers[spec.name] = det["center"]

    return visible_button_names, button_centers


def assign_buttons_after_calibration(detections, calibrated_positions):
    """
    Convert raw ArUco detections into visible button names.

    Unique marker ids are matched by id. Duplicate marker ids are matched by
    distance to their calibrated locations so that duplicate spacebar markers
    can still work if you printed the same marker twice.
    """
    detections_by_id = group_detections_by_aruco_id(detections)

    visible_button_names = set()
    button_centers = {}

    for marker_id, specs in SPECS_BY_ARUCO_ID.items():
        dets = detections_by_id.get(marker_id, [])

        if not dets:
            continue

        # Simple path for the normal case: one ArUco id per button.
        if len(specs) == 1:
            spec = specs[0]

            if spec.name in calibrated_positions:
                target = np.array(calibrated_positions[spec.name])
                det = min(
                    dets,
                    key=lambda item: np.linalg.norm(np.array(item["center"]) - target),
                )
            else:
                det = dets[0]

            visible_button_names.add(spec.name)
            button_centers[spec.name] = det["center"]
            continue

        # Duplicate-id path. Greedy nearest-neighbor matching to calibrated
        # button positions, with a distance threshold.
        unused_indices = set(range(len(dets)))
        specs_sorted = sorted(specs, key=lambda spec: BUTTON_PRIORITY.index(spec.name))

        for spec in specs_sorted:
            if spec.name not in calibrated_positions or not unused_indices:
                continue

            target = np.array(calibrated_positions[spec.name])

            best_index = None
            best_dist = None

            for idx in unused_indices:
                center = np.array(dets[idx]["center"])
                dist = float(np.linalg.norm(center - target))

                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_index = idx

            if best_index is not None and best_dist <= MATCH_MAX_DISTANCE_PX:
                unused_indices.remove(best_index)
                visible_button_names.add(spec.name)
                button_centers[spec.name] = dets[best_index]["center"]

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
):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for button_name, (cx, cy) in button_centers.items():
        button = BUTTONS_BY_NAME[button_name]
        cv2.circle(debug, (cx, cy), 18, (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"{button.label}:{button.aruco_id}",
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
            status = f"calibrating... {remaining:.1f}s"
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
    print("Starting ArUco keyboard input.")
    print(f"Show all {TOTAL_BUTTONS} buttons for {CALIBRATION_SECONDS:.1f} seconds to calibrate.")
    print(f"After calibration, cover a marker for {PRESS_SECONDS:.1f} seconds to press that key.")
    print("Caps Lock toggles uppercase output inside this script until pressed again.")
    print("Supported special buttons: Caps Lock, Enter, Backspace, and Spacebar.")
    print("Only main video window is enabled.")
    print("Press q in the video window or Ctrl+C to quit.")
    print("Configured button ids:")
    for button in BUTTONS:
        print(f"  {button.label:12s} name={button.name:12s} aruco_id={button.aruco_id}")
    print("Note: Spacebar L and Spacebar R are configured to use the same ArUco id.")
    print("The code expects to see TWO detections of that id when both spacebar markers are visible.")

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
            raw_counts = Counter(det["aruco_id"] for det in detections)

            if not calibrated:
                visible_button_names, button_centers = assign_buttons_for_calibration(detections)

                if visible_button_names == EXPECTED_BUTTON_NAMES:
                    if calibration_start is None:
                        calibration_start = now
                        print("All buttons visible. Starting calibration timer.")

                    if now - calibration_start >= CALIBRATION_SECONDS:
                        calibrated = True
                        calibrated_positions = button_centers.copy()

                        print("Calibration complete.")
                        print("Saved button positions:")

                        for button_name in BUTTON_PRIORITY:
                            button = BUTTONS_BY_NAME[button_name]
                            print(button.aruco_id, button.label, calibrated_positions[button_name])

                else:
                    calibration_start = None

                    if now - last_status_print >= 1.0:
                        missing = EXPECTED_BUTTON_NAMES - visible_button_names
                        missing_labels = [BUTTONS_BY_NAME[name].label for name in BUTTON_PRIORITY if name in missing]
                        print(f"Waiting for all buttons. Found {len(visible_button_names)}/{TOTAL_BUTTONS}.")
                        print(f"Raw detections for spacebar id {SPACEBAR_ARUCO_ID}: {raw_counts.get(SPACEBAR_ARUCO_ID, 0)}")
                        print("Missing:", ", ".join(missing_labels))
                        last_status_print = now

            else:
                visible_button_names, button_centers = assign_buttons_after_calibration(
                    detections,
                    calibrated_positions,
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
