import subprocess
import time
import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FRAMERATE = 30

# Try the common 4x4 dictionaries. Use the one that finds the most markers.
DICTIONARIES = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
]

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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


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
        return None

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
        candidates.append((score, x, y, w, h))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, x, y, w, h = candidates[0]
    return x, y, w, h


def detect_with_dictionary(gray, dict_id):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        return [], []
    return [int(x) for x in ids.flatten()], corners


def main():
    print("Point the camera at the keyboard printout.")
    print("This scanner does NOT filter by expected IDs; it prints the raw detected ArUco IDs.")
    print("Press q in the video window or Ctrl+C to quit.")

    proc = start_camera()
    last_print = 0.0

    try:
        for frame in mjpeg_frames(proc):
            display = frame.copy()
            page_box = find_white_page(frame)

            if page_box is not None:
                x, y, w, h = page_box
                roi = frame[y:y + h, x:x + w]
                offset = (x, y)
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                roi = frame
                offset = (0, 0)

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            results = []
            for name, dict_id in DICTIONARIES:
                ids, corners = detect_with_dictionary(gray, dict_id)
                results.append((len(ids), name, ids, corners))

            results.sort(reverse=True, key=lambda item: item[0])
            count, best_name, best_ids, best_corners = results[0]

            if time.time() - last_print >= 1.0:
                print("---")
                print("page found:" if page_box is not None else "page NOT found; scanning full frame")
                for c, name, ids, _ in results:
                    print(f"{name}: count={c}, ids={sorted(ids)}")
                last_print = time.time()

            ox, oy = offset
            for marker_id, corner in zip(best_ids, best_corners):
                pts = corner[0].copy()
                pts[:, 0] += ox
                pts[:, 1] += oy
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.polylines(display, [pts.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(display, str(marker_id), (cx - 15, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(display, f"Best: {best_name} count={count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("ArUco ID scanner", display)
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
