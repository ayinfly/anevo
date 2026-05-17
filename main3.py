import subprocess
import time
import cv2
import numpy as np
from evdev import UInput, ecodes as e


WIDTH = 800
HEIGHT = 450
FRAMERATE = 30

CALIBRATION_SECONDS = 3.0
PRESS_SECONDS = 0.08
RELEASE_SECONDS = 0.04

EXPECTED_IDS = set(range(26))

ID_TO_KEY = {
    0: "q", 1: "w", 2: "e", 3: "r", 4: "t", 5: "y", 6: "u", 7: "i", 8: "o", 9: "p",
    10: "a", 11: "s", 12: "d", 13: "f", 14: "g", 15: "h", 16: "j", 17: "k", 18: "l",
    19: "z", 20: "x", 21: "c", 22: "v", 23: "b", 24: "n", 25: "m",
}

KEY_TO_EVDEV = {
    "q": e.KEY_Q, "w": e.KEY_W, "e": e.KEY_E, "r": e.KEY_R, "t": e.KEY_T,
    "y": e.KEY_Y, "u": e.KEY_U, "i": e.KEY_I, "o": e.KEY_O, "p": e.KEY_P,
    "a": e.KEY_A, "s": e.KEY_S, "d": e.KEY_D, "f": e.KEY_F, "g": e.KEY_G,
    "h": e.KEY_H, "j": e.KEY_J, "k": e.KEY_K, "l": e.KEY_L,
    "z": e.KEY_Z, "x": e.KEY_X, "c": e.KEY_C, "v": e.KEY_V, "b": e.KEY_B,
    "n": e.KEY_N, "m": e.KEY_M,
}

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])
MIN_PAGE_AREA = 10000

MIN_FINGERTIP_AREA = 80
MAX_FINGERTIP_AREA = 6000
DEFAULT_KEY_RADIUS = 38

# Put a different colored sticker/tape dot on each fingertip you want to track.
# Tune these HSV ranges for your lighting by watching the debug masks.
FINGER_COLORS = {
    "red": [
        (np.array([0, 100, 80]), np.array([10, 255, 255])),
        (np.array([170, 100, 80]), np.array([180, 255, 255])),
    ],
    "yellow": [
        (np.array([18, 80, 90]), np.array([38, 255, 255])),
    ],
    "green": [
        (np.array([40, 70, 70]), np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([90, 80, 70]), np.array([130, 255, 255])),
    ],
    "purple": [
        (np.array([130, 60, 70]), np.array([165, 255, 255])),
    ],
}

FINGER_DRAW_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "purple": (255, 0, 255),
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

    visible_ids = set()
    marker_centers = {}

    if ids is not None:
        ids = ids.flatten()

        for i, marker_id in enumerate(ids):
            marker_id = int(marker_id)

            if marker_id not in EXPECTED_IDS:
                continue

            pts = corners[i][0].copy()
            pts[:, 0] += x
            pts[:, 1] += y

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            visible_ids.add(marker_id)
            marker_centers[marker_id] = (cx, cy)

    return visible_ids, marker_centers


def estimate_key_radius(marker_positions):
    nearest_distances = []
    positions = list(marker_positions.values())

    for i, (x1, y1) in enumerate(positions):
        nearest = None

        for j, (x2, y2) in enumerate(positions):
            if i == j:
                continue

            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            if nearest is None or dist < nearest:
                nearest = dist

        if nearest is not None:
            nearest_distances.append(nearest)

    if not nearest_distances:
        return DEFAULT_KEY_RADIUS

    return max(20, int(np.median(nearest_distances) * 0.45))


def mask_for_color(hsv, ranges):
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))

    combined = cv2.erode(combined, None, iterations=1)
    combined = cv2.dilate(combined, None, iterations=2)

    return combined


def detect_colored_fingertips(frame, page_box):
    x, y, w, h, page_area = page_box
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    fingertips = {}
    color_masks = {}

    for finger_name, ranges in FINGER_COLORS.items():
        mask = mask_for_color(hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_masks[finger_name] = mask

        if not contours:
            continue

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < MIN_FINGERTIP_AREA or area > MAX_FINGERTIP_AREA:
                continue

            moments = cv2.moments(contour)

            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"]) + x
            cy = int(moments["m01"] / moments["m00"]) + y
            candidates.append((area, cx, cy))

        if candidates:
            area, cx, cy = max(candidates)
            fingertips[finger_name] = (cx, cy, area)

    return fingertips, color_masks


def nearest_key_for_point(point, marker_positions, key_radius):
    px, py = point
    best_id = None
    best_dist = None

    for marker_id, (kx, ky) in marker_positions.items():
        dist = ((px - kx) ** 2 + (py - ky) ** 2) ** 0.5

        if best_dist is None or dist < best_dist:
            best_id = marker_id
            best_dist = dist

    if best_id is None or best_dist > key_radius:
        return None, best_dist

    return ID_TO_KEY[best_id], best_dist


def send_key(ui, key):
    code = KEY_TO_EVDEV[key]

    ui.write(e.EV_KEY, code, 1)
    ui.syn()

    time.sleep(0.02)

    ui.write(e.EV_KEY, code, 0)
    ui.syn()


def update_finger_states(fingertips, marker_positions, key_radius, finger_states, ui, now):
    active_fingers = set(fingertips)

    for finger_name, state in finger_states.items():
        if finger_name in active_fingers:
            continue

        if state["last_seen"] is not None and now - state["last_seen"] < RELEASE_SECONDS:
            continue

        state["key"] = None
        state["entered_at"] = None
        state["pressed"] = False

    for finger_name, (cx, cy, area) in fingertips.items():
        key, distance = nearest_key_for_point((cx, cy), marker_positions, key_radius)
        state = finger_states[finger_name]
        state["last_seen"] = now

        if key is None:
            state["key"] = None
            state["entered_at"] = None
            state["pressed"] = False
            continue

        if key != state["key"]:
            state["key"] = key
            state["entered_at"] = now
            state["pressed"] = False

        if not state["pressed"] and now - state["entered_at"] >= PRESS_SECONDS:
            print(f"Pressed {key} with {finger_name}.")
            send_key(ui, key)
            state["pressed"] = True


def draw_debug(
    frame,
    page_box,
    visible_ids,
    marker_centers,
    calibrated,
    calibration_start,
    calibrated_positions,
    key_radius,
    fingertips,
    finger_states,
):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    positions_to_draw = calibrated_positions if calibrated else marker_centers

    for marker_id, (cx, cy) in positions_to_draw.items():
        key = ID_TO_KEY[marker_id]
        cv2.circle(debug, (cx, cy), key_radius, (80, 80, 80), 1)
        cv2.circle(debug, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            debug,
            key,
            (cx - 8, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    found = len(visible_ids)

    if not calibrated:
        if found == 26 and calibration_start is not None:
            remaining = max(0, CALIBRATION_SECONDS - (time.time() - calibration_start))
            status = f"calibrating... {remaining:.1f}s"
        else:
            status = f"show all markers: {found}/26"
        color = (0, 255, 255)

    else:
        status = f"finger typing ready  fingers={len(fingertips)}  radius={key_radius}"
        color = (0, 255, 0)

    cv2.putText(
        debug,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    active_labels = []

    for finger_name, (cx, cy, area) in fingertips.items():
        draw_color = FINGER_DRAW_COLORS.get(finger_name, (255, 255, 255))
        state = finger_states[finger_name]
        label = finger_name

        if state["key"] is not None:
            label = f"{finger_name}:{state['key']}"
            active_labels.append(f"{finger_name}->{state['key']}")

        cv2.circle(debug, (cx, cy), 14, draw_color, 2)
        cv2.putText(
            debug,
            label,
            (cx + 16, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            draw_color,
            1,
            cv2.LINE_AA,
        )

    if calibrated and active_labels:
        cv2.putText(
            debug,
            " ".join(active_labels[:6]),
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def main():
    print("Starting ArUco-calibrated fingertip keyboard.")
    print("Show all 26 markers for 3 seconds to calibrate.")
    print("After calibration, use colored fingertip stickers to press keys.")
    print("Default fingertip colors: red, yellow, green, blue, purple.")
    print("Press q in the debug window to quit.")

    aruco_obj = get_aruco_detector()
    proc = start_camera()
    ui = UInput()

    calibrated = False
    calibrated_positions = {}
    key_radius = DEFAULT_KEY_RADIUS
    calibration_start = None

    finger_states = {
        finger_name: {
            "key": None,
            "entered_at": None,
            "pressed": False,
            "last_seen": None,
        }
        for finger_name in FINGER_COLORS
    }

    try:
        for frame in mjpeg_frames(proc):
            now = time.time()
            page_box, page_mask = find_white_page(frame)

            if page_box is None:
                print("No white page found.")
                calibration_start = None

                cv2.imshow("debug", frame)
                cv2.imshow("white page mask", page_mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            if not calibrated:
                visible_ids, marker_centers = detect_aruco_inside_page(frame, page_box, aruco_obj)
                fingertips = {}

                if visible_ids == EXPECTED_IDS:
                    if calibration_start is None:
                        calibration_start = now
                        print("All markers visible. Starting 3 second calibration timer.")

                    if now - calibration_start >= CALIBRATION_SECONDS:
                        calibrated = True
                        calibrated_positions = marker_centers.copy()
                        key_radius = estimate_key_radius(calibrated_positions)

                        print("Calibration complete.")
                        print(f"Estimated key radius: {key_radius}px")
                        print("Saved marker positions:")

                        for marker_id in sorted(calibrated_positions):
                            print(marker_id, ID_TO_KEY[marker_id], calibrated_positions[marker_id])

                else:
                    calibration_start = None
                    print(f"Waiting for all markers. Found {len(visible_ids)}/26.")

            else:
                visible_ids = EXPECTED_IDS
                marker_centers = {}
                fingertips, color_masks = detect_colored_fingertips(frame, page_box)
                update_finger_states(
                    fingertips=fingertips,
                    marker_positions=calibrated_positions,
                    key_radius=key_radius,
                    finger_states=finger_states,
                    ui=ui,
                    now=now,
                )

            debug = draw_debug(
                frame=frame,
                page_box=page_box,
                visible_ids=visible_ids,
                marker_centers=marker_centers,
                calibrated=calibrated,
                calibration_start=calibration_start,
                calibrated_positions=calibrated_positions,
                key_radius=key_radius,
                fingertips=fingertips,
                finger_states=finger_states,
            )

            cv2.imshow("debug", debug)
            cv2.imshow("white page mask", page_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping.")

    finally:
        proc.terminate()
        proc.wait()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
