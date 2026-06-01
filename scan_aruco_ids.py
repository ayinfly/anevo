import subprocess
import time

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
FRAMERATE = 30

# Try common 4x4 ArUco dictionaries and use the one with the most detections.
DICTIONARIES = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
]

# HSV range for detecting the white keyboard printout/page.
WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])

MIN_PAGE_AREA = 10_000


def start_camera():
    """Start the Raspberry Pi camera and stream MJPEG frames to stdout."""
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
    """Yield decoded OpenCV frames from an MJPEG byte stream."""
    buffer = b""

    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break

        buffer += chunk

        start = buffer.find(b"\xff\xd8")  # JPEG start marker
        end = buffer.find(b"\xff\xd9")    # JPEG end marker

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
    Find the most likely white keyboard printout in the frame.

    Returns:
        (x, y, w, h) if a page-like region is found, otherwise None.
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
        return None

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

        # Prefer large, centered white regions.
        score = area - distance_from_center * 20
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, x, y, w, h = candidates[0]

    return x, y, w, h


def detect_with_dictionary(gray, dict_id):
    """
    Detect ArUco markers using a specific dictionary.

    Returns:
        ids: list[int]
        corners: list of corner arrays from OpenCV
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # Support both newer and older OpenCV ArUco APIs.
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )

    if ids is None:
        return [], []

    return [int(marker_id) for marker_id in ids.flatten()], corners


def draw_detected_markers(display, marker_ids, corners, offset):
    """Draw detected marker outlines and IDs on the display frame."""
    offset_x, offset_y = offset

    for marker_id, corner in zip(marker_ids, corners):
        points = corner[0].copy()
        points[:, 0] += offset_x
        points[:, 1] += offset_y

        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        cv2.polylines(
            display,
            [points.astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            str(marker_id),
            (center_x - 15, center_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def print_detection_results(page_box, results):
    """Print detected marker IDs for each tested dictionary."""
    print("---")

    if page_box is not None:
        print("page found")
    else:
        print("page NOT found; scanning full frame")

    for count, name, ids, _ in results:
        print(f"{name}: count={count}, ids={sorted(ids)}")


def main():
    print("Point the camera at the keyboard printout.")
    print("This scanner prints raw detected ArUco IDs.")
    print("Press q in the video window or Ctrl+C to quit.")

    proc = start_camera()
    last_print_time = 0.0

    try:
        for frame in mjpeg_frames(proc):
            display = frame.copy()

            page_box = find_white_page(frame)

            if page_box is not None:
                x, y, w, h = page_box
                roi = frame[y:y + h, x:x + w]
                offset = (x, y)

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2,
                )
            else:
                roi = frame
                offset = (0, 0)

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            results = []

            for name, dict_id in DICTIONARIES:
                marker_ids, corners = detect_with_dictionary(gray, dict_id)
                results.append((len(marker_ids), name, marker_ids, corners))

            results.sort(reverse=True, key=lambda item: item[0])

            best_count, best_name, best_ids, best_corners = results[0]

            if time.time() - last_print_time >= 1.0:
                print_detection_results(page_box, results)
                last_print_time = time.time()

            draw_detected_markers(
                display,
                best_ids,
                best_corners,
                offset,
            )

            cv2.putText(
                display,
                f"Best: {best_name} count={best_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

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