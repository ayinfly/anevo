# we used chatgpt to generate the new updates
# generated better mapping
# fact checked and made sure everything was working correctly

import subprocess
import time
from dataclasses import dataclass
from collections import Counter, defaultdict

import cv2
import numpy as np
from evdev import UInput, ecodes as e


WIDTH = 640
HEIGHT = 360
FRAMERATE = 30

CALIBRATION_SECONDS = 3.0
PRESS_SECONDS = 0.1

# Used only for matching duplicate IDs like the two spacebar markers.
# Coordinates are normalized inside the detected page box, so this is robust to
# the page moving closer/farther or shifting in the camera view.
DUPLICATE_MATCH_MAX_NORM_DIST = 0.20

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])
MIN_PAGE_AREA = 10000


@dataclass(frozen=True)
class ButtonSpec:
    name: str
    aruco_id: int
    action: str
    label: str


# Hardcoded from the uploaded printout:
#
# Top row:
#   Q=0 W=1 E=2 R=3 T=4 Y=5 U=6 I=7 O=8 P=9 Backspace=10
# Middle row:
#   Caps=11 A=12 S=13 D=14 F=15 G=16 H=17 J=18 K=19 L=20 Enter=21
# Bottom row:
#   Z=22 X=23 C=24 V=25 B=26 N=27 M=28
# Spacebar:
#   both space markers use ID 29
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
    ButtonSpec("backspace", 10, "backspace", "Backspace"),

    ButtonSpec("caps_lock", 11, "caps_lock", "Caps Lock"),
    ButtonSpec("a", 12, "a", "A"),
    ButtonSpec("s", 13, "s", "S"),
    ButtonSpec("d", 14, "d", "D"),
    ButtonSpec("f", 15, "f", "F"),
    ButtonSpec("g", 16, "g", "G"),
    ButtonSpec("h", 17, "h", "H"),
    ButtonSpec("j", 18, "j", "J"),
    ButtonSpec("k", 19, "k", "K"),
    ButtonSpec("l", 20, "l", "L"),
    ButtonSpec("enter", 21, "enter", "Enter"),

    ButtonSpec("z", 22, "z", "Z"),
    ButtonSpec("x", 23, "x", "X"),
    ButtonSpec("c", 24, "c", "C"),
    ButtonSpec("v", 25, "v", "V"),
    ButtonSpec("b", 26, "b", "B"),
    ButtonSpec("n", 27, "n", "N"),
    ButtonSpec("m", 28, "m", "M"),

    ButtonSpec("space_left", 29, "space", "Spacebar L"),
    ButtonSpec("space_right", 29, "space", "Spacebar R"),
]

BUTTON_BY_NAME = {button.name: button for button in BUTTONS}
BUTTONS_BY_ARUCO_ID = defaultdict(list)
for button in BUTTONS:
    BUTTONS_BY_ARUCO_ID[button.aruco_id].append(button)

EXPECTED_IDS = set(BUTTONS_BY_ARUCO_ID.keys())
EXPECTED_ID_COUNTS = Counter(button.aruco_id for button in BUTTONS)

# This is the order used when multiple markers are covered at once.
BUTTON_PRIORITY = [button.name for button in BUTTONS]

KEY_TO_EVDEV = {
    "q": e.KEY_Q, "w": e.KEY_W, "e": e.KEY_E, "r": e.KEY_R, "t": e.KEY_T,
    "y": e.KEY_Y, "u": e.KEY_U, "i": e.KEY_I, "o": e.KEY_O, "p": e.KEY_P,
    "a": e.KEY_A, "s": e.KEY_S, "d": e.KEY_D, "f": e.KEY_F, "g": e.KEY_G,
    "h": e.KEY_H, "j": e.KEY_J, "k": e.KEY_K, "l": e.KEY_L,
    "z": e.KEY_Z, "x": e.KEY_X, "c": e.KEY_C, "v": e.KEY_V, "b": e.KEY_B,
    "n": e.KEY_N, "m": e.KEY_M,
    "space": e.KEY_SPACE,
    "enter": e.KEY_ENTER,
    "backspace": e.KEY_BACKSPACE,
}


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
    # The uploaded printout uses OpenCV's 4x4 ArUco dictionary with IDs 0-29.
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

            if marker_id not in EXPECTED_IDS:
                continue

            pts = corners[i][0].copy()
            pts[:, 0] += x
            pts[:, 1] += y

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            norm_cx = (cx - x) / max(w, 1)
            norm_cy = (cy - y) / max(h, 1)

            detections.append({
                "id": marker_id,
                "center": (int(cx), int(cy)),
                "norm_center": (float(norm_cx), float(norm_cy)),
                "corners": pts,
            })

    return detections


def detection_counts_by_id(detections):
    return Counter(d["id"] for d in detections)


def has_all_expected_markers(detections):
    counts = detection_counts_by_id(detections)

    for marker_id, required_count in EXPECTED_ID_COUNTS.items():
        if counts[marker_id] < required_count:
            return False

    return True


def assign_calibrated_positions(detections):
    """
    Saves each physical button's normalized page location.

    For unique marker IDs, this is direct.
    For duplicated IDs, such as the two spacebar markers with ID 29, detections
    are sorted left-to-right and assigned to space_left / space_right.
    """
    detections_by_id = defaultdict(list)

    for d in detections:
        detections_by_id[d["id"]].append(d)

    calibrated_positions = {}

    for marker_id, buttons in BUTTONS_BY_ARUCO_ID.items():
        matches = detections_by_id[marker_id]
        matches = sorted(matches, key=lambda d: d["norm_center"][0])
        buttons_sorted = sorted(buttons, key=lambda b: BUTTON_PRIORITY.index(b.name))

        for button, detection in zip(buttons_sorted, matches):
            calibrated_positions[button.name] = detection["norm_center"]

    return calibrated_positions


def match_visible_buttons(detections, calibrated_positions):
    """
    Returns:
      visible_buttons: set of physical button names visible right now
      button_centers: dict from physical button name -> current pixel center

    For unique marker IDs, any detection of that ID means the button is visible.
    For duplicate marker IDs, detections are matched to the calibrated left/right
    slots by nearest normalized position.
    """
    visible_buttons = set()
    button_centers = {}

    detections_by_id = defaultdict(list)

    for d in detections:
        detections_by_id[d["id"]].append(d)

    for marker_id, buttons in BUTTONS_BY_ARUCO_ID.items():
        matches = detections_by_id.get(marker_id, [])

        if not matches:
            continue

        if len(buttons) == 1:
            button = buttons[0]
            d = matches[0]
            visible_buttons.add(button.name)
            button_centers[button.name] = d["center"]
            continue

        # Duplicate IDs: greedily match each detection to its closest calibrated
        # physical button slot, without using the same slot twice.
        unmatched_buttons = set(button.name for button in buttons)

        costs = []
        for d_index, d in enumerate(matches):
            dx, dy = d["norm_center"]

            for button in buttons:
                if button.name not in calibrated_positions:
                    continue

                bx, by = calibrated_positions[button.name]
                dist = ((dx - bx) ** 2 + (dy - by) ** 2) ** 0.5
                costs.append((dist, d_index, button.name))

        costs.sort()

        used_detections = set()

        for dist, d_index, button_name in costs:
            if dist > DUPLICATE_MATCH_MAX_NORM_DIST:
                continue

            if d_index in used_detections or button_name not in unmatched_buttons:
                continue

            used_detections.add(d_index)
            unmatched_buttons.remove(button_name)
            visible_buttons.add(button_name)
            button_centers[button_name] = matches[d_index]["center"]

    return visible_buttons, button_centers


def get_frontmost_missing_button(missing_buttons):
    for button_name in BUTTON_PRIORITY:
        if button_name in missing_buttons:
            return button_name

    return None


def tap_key(ui, code):
    ui.write(e.EV_KEY, code, 1)
    ui.syn()

    time.sleep(0.04)

    ui.write(e.EV_KEY, code, 0)
    ui.syn()


def tap_shifted_key(ui, code):
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1)
    ui.syn()

    tap_key(ui, code)

    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 0)
    ui.syn()


def press_button(ui, button, caps_lock_on):
    action = button.action

    if action == "caps_lock":
        caps_lock_on = not caps_lock_on
        print(f"Caps Lock toggled {'ON' if caps_lock_on else 'OFF'}.")
        return caps_lock_on

    code = KEY_TO_EVDEV[action]

    if action in "abcdefghijklmnopqrstuvwxyz" and caps_lock_on:
        tap_shifted_key(ui, code)
    else:
        tap_key(ui, code)

    return caps_lock_on


def draw_debug(
    frame,
    page_box,
    detections,
    visible_buttons,
    button_centers,
    calibrated,
    calibration_start,
    active_button_name,
    active_missing_start,
    already_pressed,
    caps_lock_on,
    fps,
):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw raw detections in gray/white first.
    for d in detections:
        marker_id = d["id"]
        cx, cy = d["center"]
        cv2.circle(debug, (cx, cy), 12, (200, 200, 200), 1)
        cv2.putText(
            debug,
            f"id:{marker_id}",
            (cx - 20, cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Draw matched physical buttons in green.
    for button_name, (cx, cy) in button_centers.items():
        button = BUTTON_BY_NAME[button_name]
        cv2.circle(debug, (cx, cy), 18, (0, 255, 0), 2)
        cv2.putText(
            debug,
            button.label,
            (cx - 35, cy + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    found_buttons = len(visible_buttons)
    missing_buttons = set(BUTTON_BY_NAME.keys()) - visible_buttons

    if not calibrated:
        total_raw = len(detections)
        total_needed = len(BUTTONS)
        if has_all_expected_markers(detections) and calibration_start is not None:
            remaining = max(0, CALIBRATION_SECONDS - (time.time() - calibration_start))
            status = f"calibrating... {remaining:.1f}s"
        else:
            counts = detection_counts_by_id(detections)
            space_count = counts[29]
            status = f"show all markers: raw={total_raw}/{total_needed} space={space_count}/2"
        color = (0, 255, 255)

    else:
        status = f"ready found={found_buttons}/{len(BUTTONS)} missing={len(missing_buttons)}"
        color = (0, 255, 0) if len(missing_buttons) == 0 else (0, 0, 255)

    cv2.putText(
        debug,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
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

    caps_text = f"CAPS LOCK: {'ON' if caps_lock_on else 'OFF'}"
    caps_color = (0, 255, 255) if caps_lock_on else (180, 180, 180)
    cv2.putText(
        debug,
        caps_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        caps_color,
        2,
        cv2.LINE_AA,
    )

    if calibrated and active_button_name is not None:
        button = BUTTON_BY_NAME[active_button_name]
        elapsed = 0.0

        if active_missing_start is not None:
            elapsed = time.time() - active_missing_start

        press_status = "pressed" if active_button_name in already_pressed else "timing"

        cv2.putText(
            debug,
            f"frontmost missing: {button.label} {elapsed:.2f}s {press_status}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def print_expected_mapping():
    print("Using hardcoded mapping from the uploaded printout:")
    for button in BUTTONS:
        print(f"  id={button.aruco_id:2d} -> {button.label}")


def main():
    print("Starting ArUco keyboard input.")
    print_expected_mapping()
    print()
    print(f"Show all {len(BUTTONS)} physical markers for {CALIBRATION_SECONDS:.1f} seconds to calibrate.")
    print("Note: both spacebar markers use ArUco ID 29, so calibration needs to see ID 29 twice.")
    print(f"After calibration, cover a marker for {PRESS_SECONDS:.1f} seconds to press that key.")
    print("Caps Lock toggles internal caps mode. When caps is ON, letters are sent as Shift+letter.")
    print("Press q in the video window or Ctrl+C to quit.")

    aruco_obj = get_aruco_detector()
    proc = start_camera()

    key_codes = set(KEY_TO_EVDEV.values())
    key_codes.add(e.KEY_LEFTSHIFT)

    capabilities = {
        e.EV_KEY: list(key_codes)
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

            visible_buttons = set()
            button_centers = {}

            if not calibrated:
                if has_all_expected_markers(detections):
                    if calibration_start is None:
                        calibration_start = now
                        print("All physical markers visible. Starting calibration timer.")

                    if now - calibration_start >= CALIBRATION_SECONDS:
                        calibrated = True
                        calibrated_positions = assign_calibrated_positions(detections)
                        visible_buttons, button_centers = match_visible_buttons(
                            detections,
                            calibrated_positions,
                        )

                        print("Calibration complete.")
                        print("Saved physical button positions:")

                        for button in BUTTONS:
                            pos = calibrated_positions.get(button.name)
                            print(f"  {button.label:12s} id={button.aruco_id:2d} pos={pos}")

                else:
                    calibration_start = None

                    if now - last_status_print >= 1.0:
                        counts = detection_counts_by_id(detections)
                        missing_parts = []
                        for marker_id, required_count in sorted(EXPECTED_ID_COUNTS.items()):
                            found_count = counts[marker_id]
                            if found_count < required_count:
                                missing_parts.append(f"id {marker_id}: {found_count}/{required_count}")

                        missing_text = ", ".join(missing_parts[:8])
                        if len(missing_parts) > 8:
                            missing_text += ", ..."

                        print(
                            f"Waiting for all markers. "
                            f"Raw found {len(detections)}/{len(BUTTONS)}. "
                            f"Missing/low: {missing_text}"
                        )
                        last_status_print = now

            else:
                visible_buttons, button_centers = match_visible_buttons(
                    detections,
                    calibrated_positions,
                )

                missing_buttons = set(BUTTON_BY_NAME.keys()) - visible_buttons
                frontmost_missing_button = get_frontmost_missing_button(missing_buttons)

                if frontmost_missing_button is None:
                    active_button_name = None
                    active_missing_start = None

                else:
                    if frontmost_missing_button != active_button_name:
                        active_button_name = frontmost_missing_button
                        active_missing_start = now

                    button = BUTTON_BY_NAME[active_button_name]
                    missing_time = now - active_missing_start

                    if missing_time >= PRESS_SECONDS and active_button_name not in already_pressed:
                        print(f"Pressed {button.label}. Missing for {missing_time:.2f}s.")
                        caps_lock_on = press_button(ui, button, caps_lock_on)
                        already_pressed.add(active_button_name)

                visible_again = set(already_pressed) & visible_buttons

                for button_name in visible_again:
                    button = BUTTON_BY_NAME[button_name]
                    print(f"{button.label} visible again. Can press again.")
                    already_pressed.remove(button_name)

            debug = draw_debug(
                frame=frame,
                page_box=page_box,
                detections=detections,
                visible_buttons=visible_buttons,
                button_centers=button_centers,
                calibrated=calibrated,
                calibration_start=calibration_start,
                active_button_name=active_button_name,
                active_missing_start=active_missing_start,
                already_pressed=already_pressed,
                caps_lock_on=caps_lock_on,
                fps=fps,
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
