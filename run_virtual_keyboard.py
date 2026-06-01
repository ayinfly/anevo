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
KEY_HOLD_SECONDS = 0.025

# Reuse the detected page box most frames for speed, but rescan sometimes
# in case the page moves.
PAGE_RESCAN_EVERY_FRAMES = 10
PAGE_BOX_PADDING_RATIO = 0.04
MIN_TRACKED_DETECTIONS = 20

# Used for duplicated IDs, such as the two spacebar markers.
# Coordinates are normalized within the page box.
DUPLICATE_MATCH_MAX_NORM_DIST = 0.20

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])
MIN_PAGE_AREA = 10_000


@dataclass(frozen=True)
class ButtonSpec:
    """Represents one physical button on the printed keyboard."""

    name: str
    aruco_id: int
    action: str
    label: str


# Hardcoded from the keyboard printout.
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

    # Both spacebar markers intentionally use the same ArUco ID.
    ButtonSpec("space_left", 29, "space", "Spacebar L"),
    ButtonSpec("space_right", 29, "space", "Spacebar R"),
]

BUTTON_BY_NAME = {button.name: button for button in BUTTONS}

SPACE_BUTTON_NAMES = {
    button.name
    for button in BUTTONS
    if button.action == "space"
}

BUTTONS_BY_ARUCO_ID = defaultdict(list)
for button in BUTTONS:
    BUTTONS_BY_ARUCO_ID[button.aruco_id].append(button)

EXPECTED_IDS = set(BUTTONS_BY_ARUCO_ID.keys())
EXPECTED_ID_COUNTS = Counter(button.aruco_id for button in BUTTONS)

# If multiple markers are covered, this order decides which one is typed.
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
    "space": e.KEY_SPACE,
    "enter": e.KEY_ENTER,
    "backspace": e.KEY_BACKSPACE,
}


def start_camera():
    """Start the Raspberry Pi camera as an MJPEG stream."""
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
    """Yield decoded OpenCV frames from the camera process."""
    buffer = b""

    while True:
        chunk = proc.stdout.read(4096)

        if not chunk:
            break

        buffer += chunk

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")

        if start == -1 or end == -1 or end <= start:
            continue

        jpg = buffer[start:end + 2]
        buffer = buffer[end + 2:]

        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is not None:
            yield frame


def find_white_page(frame):
    """
    Find the most likely white keyboard page.

    Returns:
        page_box: (x, y, w, h, area), or None
        mask: HSV threshold mask, useful for debugging
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, mask

    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_PAGE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        page_center_x = x + w / 2
        page_center_y = y + h / 2

        distance_from_center = (
            (page_center_x - frame_center_x) ** 2
            + (page_center_y - frame_center_y) ** 2
        ) ** 0.5

        # Prefer large white regions near the center of the camera view.
        score = area - distance_from_center * 20
        candidates.append((score, x, y, w, h, area))

    if not candidates:
        return None, mask

    candidates.sort(reverse=True)
    _, x, y, w, h, area = candidates[0]

    return (x, y, w, h, area), mask


def pad_page_box(page_box, frame_shape):
    """Add padding around the detected page box while staying inside the frame."""
    x, y, w, h, area = page_box
    frame_h, frame_w = frame_shape[:2]

    pad = int(max(w, h) * PAGE_BOX_PADDING_RATIO)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)

    return (x1, y1, x2 - x1, y2 - y1, area)


def get_aruco_detector():
    """Create an ArUco detector for the keyboard printout's dictionary."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Support both newer and older OpenCV ArUco APIs.
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return dictionary, detector

    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_aruco_inside_page(frame, page_box, aruco_obj):
    """
    Detect expected ArUco markers inside the current page box.

    Each detection includes pixel coordinates and normalized page coordinates.
    Normalized coordinates make matching robust when the page moves or resizes.
    """
    x, y, w, h, page_area = page_box

    roi = frame[y:y + h, x:x + w]
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

    if ids is None:
        return detections

    for i, marker_id in enumerate(ids.flatten()):
        marker_id = int(marker_id)

        if marker_id not in EXPECTED_IDS:
            continue

        points = corners[i][0].copy()
        points[:, 0] += x
        points[:, 1] += y

        center_x = float(np.mean(points[:, 0]))
        center_y = float(np.mean(points[:, 1]))

        norm_center_x = (center_x - x) / max(w, 1)
        norm_center_y = (center_y - y) / max(h, 1)

        detections.append({
            "id": marker_id,
            "center": (int(center_x), int(center_y)),
            "norm_center": (float(norm_center_x), float(norm_center_y)),
            "corners": points,
        })

    return detections


def detection_counts_by_id(detections):
    """Count how many times each expected ArUco ID was detected."""
    return Counter(detection["id"] for detection in detections)


def has_all_expected_markers(detections):
    """Return True only if every physical button marker is visible."""
    counts = detection_counts_by_id(detections)

    for marker_id, required_count in EXPECTED_ID_COUNTS.items():
        if counts[marker_id] < required_count:
            return False

    return True


def assign_calibrated_positions(detections):
    """
    Save each physical button's normalized page position.

    Unique marker IDs map directly to one button. Duplicate marker IDs are sorted
    left-to-right, which is enough for the two spacebar markers.
    """
    detections_by_id = defaultdict(list)

    for detection in detections:
        detections_by_id[detection["id"]].append(detection)

    calibrated_positions = {}

    for marker_id, buttons in BUTTONS_BY_ARUCO_ID.items():
        matches = sorted(
            detections_by_id[marker_id],
            key=lambda detection: detection["norm_center"][0],
        )
        buttons_sorted = sorted(
            buttons,
            key=lambda button: BUTTON_PRIORITY.index(button.name),
        )

        for button, detection in zip(buttons_sorted, matches):
            calibrated_positions[button.name] = detection["norm_center"]

    return calibrated_positions


def match_visible_buttons(detections, calibrated_positions):
    """
    Match current detections to physical button names.

    Returns:
        visible_buttons: set[str]
        button_centers: dict[str, tuple[int, int]]
    """
    visible_buttons = set()
    button_centers = {}

    detections_by_id = defaultdict(list)

    for detection in detections:
        detections_by_id[detection["id"]].append(detection)

    for marker_id, buttons in BUTTONS_BY_ARUCO_ID.items():
        matches = detections_by_id.get(marker_id, [])

        if not matches:
            continue

        if len(buttons) == 1:
            button = buttons[0]
            detection = matches[0]

            visible_buttons.add(button.name)
            button_centers[button.name] = detection["center"]
            continue

        # Duplicate IDs need position-based matching.
        unmatched_buttons = set(button.name for button in buttons)
        costs = []

        for detection_index, detection in enumerate(matches):
            detection_x, detection_y = detection["norm_center"]

            for button in buttons:
                if button.name not in calibrated_positions:
                    continue

                button_x, button_y = calibrated_positions[button.name]

                distance = (
                    (detection_x - button_x) ** 2
                    + (detection_y - button_y) ** 2
                ) ** 0.5

                costs.append((distance, detection_index, button.name))

        costs.sort()

        used_detections = set()

        for distance, detection_index, button_name in costs:
            if distance > DUPLICATE_MATCH_MAX_NORM_DIST:
                continue

            if detection_index in used_detections:
                continue

            if button_name not in unmatched_buttons:
                continue

            used_detections.add(detection_index)
            unmatched_buttons.remove(button_name)

            visible_buttons.add(button_name)
            button_centers[button_name] = matches[detection_index]["center"]

    return visible_buttons, button_centers


def apply_spacebar_press_rule(missing_buttons):
    """
    Avoid accidental spacebar presses when only one space marker is covered.

    Because the spacebar has two markers, require both to be missing before
    treating the spacebar as pressed.
    """
    missing_space_buttons = missing_buttons & SPACE_BUTTON_NAMES

    if missing_space_buttons and missing_space_buttons != SPACE_BUTTON_NAMES:
        return missing_buttons - missing_space_buttons

    return missing_buttons


def get_frontmost_missing_button(missing_buttons):
    """Choose the highest-priority covered button."""
    for button_name in BUTTON_PRIORITY:
        if button_name in missing_buttons:
            return button_name

    return None


def release_due_keys(ui, pending_releases, now):
    """Release any virtual keys whose hold time has expired."""
    due_codes = [
        code
        for code, release_at in pending_releases.items()
        if now >= release_at
    ]

    for code in due_codes:
        ui.write(e.EV_KEY, code, 0)
        del pending_releases[code]

    if due_codes:
        ui.syn()


def hold_key(ui, pending_releases, code, now):
    """Press a key now and schedule its release."""
    ui.write(e.EV_KEY, code, 1)
    pending_releases[code] = now + KEY_HOLD_SECONDS
    ui.syn()


def hold_shifted_key(ui, pending_releases, code, now):
    """Press Shift+key now and schedule both releases."""
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1)
    ui.write(e.EV_KEY, code, 1)

    release_at = now + KEY_HOLD_SECONDS
    pending_releases[e.KEY_LEFTSHIFT] = release_at
    pending_releases[code] = release_at

    ui.syn()


def press_button(ui, pending_releases, button, caps_lock_on, now):
    """
    Send the virtual key event for a button.

    Caps Lock is handled internally instead of sending the real OS caps key.
    """
    action = button.action

    if action == "caps_lock":
        caps_lock_on = not caps_lock_on
        print(f"Caps Lock toggled {'ON' if caps_lock_on else 'OFF'}.")
        return caps_lock_on

    code = KEY_TO_EVDEV[action]

    if action in "abcdefghijklmnopqrstuvwxyz" and caps_lock_on:
        hold_shifted_key(ui, pending_releases, code, now)
    else:
        hold_key(ui, pending_releases, code, now)

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
    """Draw the camera view with marker, calibration, and typing status."""
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for detection in detections:
        marker_id = detection["id"]
        center_x, center_y = detection["center"]

        cv2.circle(debug, (center_x, center_y), 12, (200, 200, 200), 1)
        cv2.putText(
            debug,
            f"id:{marker_id}",
            (center_x - 20, center_y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    for button_name, (center_x, center_y) in button_centers.items():
        button = BUTTON_BY_NAME[button_name]

        cv2.circle(debug, (center_x, center_y), 18, (0, 255, 0), 2)
        cv2.putText(
            debug,
            button.label,
            (center_x - 35, center_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    found_buttons = len(visible_buttons)
    missing_buttons = apply_spacebar_press_rule(
        set(BUTTON_BY_NAME.keys()) - visible_buttons
    )

    if not calibrated:
        raw_count = len(detections)
        required_count = len(BUTTONS)

        if has_all_expected_markers(detections) and calibration_start is not None:
            remaining = max(0, CALIBRATION_SECONDS - (time.time() - calibration_start))
            status = f"calibrating... {remaining:.1f}s"
        else:
            counts = detection_counts_by_id(detections)
            space_count = counts[29]
            status = f"show all markers: raw={raw_count}/{required_count} space={space_count}/2"

        color = (0, 255, 255)

    else:
        status = (
            f"ready found={found_buttons}/{len(BUTTONS)} "
            f"missing={len(missing_buttons)}"
        )
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
    """Print the ArUco ID to button mapping used by this program."""
    print("Using hardcoded mapping from the uploaded printout:")

    for button in BUTTONS:
        print(f"  id={button.aruco_id:2d} -> {button.label}")


def print_missing_marker_status(detections):
    """Print which expected marker IDs are not currently visible."""
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


def create_virtual_keyboard():
    """Create the evdev virtual keyboard used to send key presses."""
    key_codes = set(KEY_TO_EVDEV.values())
    key_codes.add(e.KEY_LEFTSHIFT)

    capabilities = {
        e.EV_KEY: list(key_codes),
    }

    return UInput(capabilities, name="aruco-keyboard")


def clean_up(ui, proc, pending_releases):
    """Release held keys and close hardware/software resources."""
    for code in list(pending_releases):
        ui.write(e.EV_KEY, code, 0)

    if pending_releases:
        ui.syn()

    ui.close()
    proc.terminate()
    proc.wait()
    cv2.destroyAllWindows()


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
    ui = create_virtual_keyboard()

    calibrated = False
    calibrated_positions = {}

    tracked_page_box = None
    frame_number = 0

    calibration_start = None

    active_button_name = None
    active_missing_start = None
    already_pressed = set()

    pending_releases = {}
    caps_lock_on = False

    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    last_status_print = 0.0

    try:
        for frame in mjpeg_frames(proc):
            now = time.time()
            frame_number += 1

            release_due_keys(ui, pending_releases, now)

            fps_frames += 1
            if now - fps_start >= 1.0:
                fps = fps_frames / (now - fps_start)
                print(f"FPS: {fps:.1f} | CAPS LOCK: {'ON' if caps_lock_on else 'OFF'}")

                fps_start = now
                fps_frames = 0

            if calibrated and tracked_page_box is not None:
                should_rescan_page = frame_number % PAGE_RESCAN_EVERY_FRAMES == 0

                if should_rescan_page:
                    page_box, page_mask = find_white_page(frame)

                    if page_box is not None:
                        tracked_page_box = pad_page_box(page_box, frame.shape)
                    else:
                        page_box = tracked_page_box
                else:
                    page_box = tracked_page_box
                    page_mask = None

            else:
                page_box, page_mask = find_white_page(frame)

                if page_box is not None:
                    tracked_page_box = pad_page_box(page_box, frame.shape)

            if page_box is None:
                calibration_start = None
                active_button_name = None
                active_missing_start = None
                tracked_page_box = None

                if now - last_status_print >= 1.0:
                    print("No white page found.")
                    last_status_print = now

                cv2.imshow("keyboard view", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            detections = detect_aruco_inside_page(frame, page_box, aruco_obj)

            # If tracking gets weak, rescan the full frame for a better page box.
            if calibrated and len(detections) < MIN_TRACKED_DETECTIONS:
                fresh_page_box, page_mask = find_white_page(frame)

                if fresh_page_box is not None:
                    tracked_page_box = pad_page_box(fresh_page_box, frame.shape)
                    page_box = tracked_page_box
                    detections = detect_aruco_inside_page(frame, page_box, aruco_obj)

            visible_buttons = set()
            button_centers = {}

            if not calibrated:
                if has_all_expected_markers(detections):
                    if calibration_start is None:
                        calibration_start = now
                        print("All physical markers visible. Starting calibration timer.")

                    if now - calibration_start >= CALIBRATION_SECONDS:
                        tracked_page_box = pad_page_box(page_box, frame.shape)

                        tracking_detections = detect_aruco_inside_page(
                            frame,
                            tracked_page_box,
                            aruco_obj,
                        )

                        if has_all_expected_markers(tracking_detections):
                            calibrated = True
                            detections = tracking_detections
                            calibrated_positions = assign_calibrated_positions(detections)

                            visible_buttons, button_centers = match_visible_buttons(
                                detections,
                                calibrated_positions,
                            )

                            print("Calibration complete.")
                            print("Saved physical button positions:")

                            for button in BUTTONS:
                                position = calibrated_positions.get(button.name)
                                print(
                                    f"  {button.label:12s} "
                                    f"id={button.aruco_id:2d} "
                                    f"pos={position}"
                                )

                        else:
                            calibration_start = None
                            tracked_page_box = None
                            print("Calibration retry: tracking box did not see every marker.")

                else:
                    calibration_start = None

                    if now - last_status_print >= 1.0:
                        print_missing_marker_status(detections)
                        last_status_print = now

            else:
                visible_buttons, button_centers = match_visible_buttons(
                    detections,
                    calibrated_positions,
                )

                missing_buttons = apply_spacebar_press_rule(
                    set(BUTTON_BY_NAME.keys()) - visible_buttons
                )

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

                    if (
                        missing_time >= PRESS_SECONDS
                        and active_button_name not in already_pressed
                    ):
                        print(f"Pressed {button.label}. Missing for {missing_time:.2f}s.")

                        caps_lock_on = press_button(
                            ui,
                            pending_releases,
                            button,
                            caps_lock_on,
                            now,
                        )

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
        clean_up(ui, proc, pending_releases)


if __name__ == "__main__":
    main()