import subprocess
import time
import cv2
import numpy as np
from evdev import UInput, ecodes as e


WIDTH = 1280
HEIGHT = 720
FRAMERATE = 15

CALIBRATION_SECONDS = 3.0
PRESS_SECONDS = 0.5

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


def send_key(ui, key):
    code = KEY_TO_EVDEV[key]

    ui.write(e.EV_KEY, code, 1)
    ui.syn()

    time.sleep(0.04)

    ui.write(e.EV_KEY, code, 0)
    ui.syn()


def draw_debug(frame, page_box, visible_ids, marker_centers, calibrated, calibration_start, missing_start):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for marker_id, (cx, cy) in marker_centers.items():
        key = ID_TO_KEY[marker_id]
        cv2.circle(debug, (cx, cy), 18, (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"{key}:{marker_id}",
            (cx - 20, cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    found = len(visible_ids)
    missing_ids = EXPECTED_IDS - visible_ids

    if not calibrated:
        if found == 26 and calibration_start is not None:
            remaining = max(0, CALIBRATION_SECONDS - (time.time() - calibration_start))
            status = f"calibrating... {remaining:.1f}s"
        else:
            status = f"show all markers: {found}/26"
        color = (0, 255, 255)
    else:
        status = f"ready found={found}/26 missing={len(missing_ids)}"
        color = (0, 255, 0) if len(missing_ids) == 0 else (0, 0, 255)

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

    if calibrated and missing_ids:
        y = 80
        for marker_id in sorted(missing_ids):
            key = ID_TO_KEY[marker_id]
            elapsed = 0

            if marker_id in missing_start:
                elapsed = time.time() - missing_start[marker_id]

            cv2.putText(
                debug,
                f"missing {key}: {elapsed:.2f}s",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

    return debug


def main():
    print("Starting ArUco keyboard input.")
    print("Show all 26 markers for 3 seconds to calibrate.")
    print("After calibration, cover a marker for 0.5 seconds to press that key.")
    print("Press q in the debug window to quit.")

    aruco_obj = get_aruco_detector()
    proc = start_camera()
    ui = UInput()

    calibrated = False
    calibrated_positions = {}

    calibration_start = None
    missing_start = {}
    already_pressed = set()

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

            visible_ids, marker_centers = detect_aruco_inside_page(frame, page_box, aruco_obj)

            if not calibrated:
                if visible_ids == EXPECTED_IDS:
                    if calibration_start is None:
                        calibration_start = now
                        print("All markers visible. Starting 3 second calibration timer.")

                    elapsed = now - calibration_start

                    if elapsed >= CALIBRATION_SECONDS:
                        calibrated = True
                        calibrated_positions = marker_centers.copy()
                        print("Calibration complete.")
                        print("Saved marker positions:")
                        for marker_id in sorted(calibrated_positions):
                            print(marker_id, ID_TO_KEY[marker_id], calibrated_positions[marker_id])
                else:
                    calibration_start = None
                    print(f"Waiting for all markers. Found {len(visible_ids)}/26.")

            else:
                missing_ids = EXPECTED_IDS - visible_ids

                for marker_id in EXPECTED_IDS:
                    key = ID_TO_KEY[marker_id]

                    if marker_id in missing_ids:
                        if marker_id not in missing_start:
                            missing_start[marker_id] = now

                        missing_time = now - missing_start[marker_id]

                        if missing_time >= PRESS_SECONDS and marker_id not in already_pressed:
                            print(f"Pressed {key}. Missing for {missing_time:.2f}s.")
                            send_key(ui, key)
                            already_pressed.add(marker_id)

                    else:
                        if marker_id in missing_start:
                            del missing_start[marker_id]

                        if marker_id in already_pressed:
                            already_pressed.remove(marker_id)

            debug = draw_debug(
                frame,
                page_box,
                visible_ids,
                marker_centers,
                calibrated,
                calibration_start,
                missing_start,
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