import subprocess
import time
import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

EXPECTED_DOTS = 26

# Bright/highlighter pink HSV range
PINK_LOWER = np.array([35, 50, 50])
PINK_UPPER = np.array([90, 255, 255])

# White paper HSV range
WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 80, 255])

MIN_DOT_AREA = 20
MAX_DOT_AREA = 5000

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


def pink_mask_inside_page(frame, page_box):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_pink_mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)

    page_only_mask = np.zeros_like(full_pink_mask)

    x, y, w, h, area = page_box
    page_only_mask[y:y+h, x:x+w] = full_pink_mask[y:y+h, x:x+w]

    page_only_mask = cv2.erode(page_only_mask, None, iterations=1)
    page_only_mask = cv2.dilate(page_only_mask, None, iterations=2)

    return page_only_mask


def find_pink_dots_inside_page(frame, page_box):
    mask = pink_mask_inside_page(frame, page_box)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []

    for c in contours:
        area = cv2.contourArea(c)

        if not (MIN_DOT_AREA < area < MAX_DOT_AREA):
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        dots.append((cx, cy, area))

    dots.sort(key=lambda p: (p[1], p[0]))

    return dots, mask


def draw_debug(frame, page_box, dots):
    debug = frame.copy()

    if page_box is not None:
        x, y, w, h, area = page_box

        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            debug,
            f"page area={area:.0f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for i, (x, y, area) in enumerate(dots):
        cv2.circle(debug, (x, y), 15, (255, 0, 255), 2)
        cv2.putText(
            debug,
            f"{i}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    found = len(dots)
    missing = max(0, EXPECTED_DOTS - found)

    if missing == 0:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(
        debug,
        f"found={found}/{EXPECTED_DOTS} missing={missing}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    return debug


def main():
    print("Starting pink dot detector.")
    print("Expected dots:", EXPECTED_DOTS)
    print("Press q in the debug window to quit.")

    proc = start_camera()

    frame_count = 0
    start_time = time.time()

    try:
        for frame in mjpeg_frames(proc):
            frame_count += 1

            page_box, page_mask = find_white_page(frame)

            if page_box is None:
                print("No white page found.")

                debug = frame.copy()
                cv2.putText(
                    debug,
                    "No white page found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("debug", debug)
                cv2.imshow("white page mask", page_mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            dots, pink_mask = find_pink_dots_inside_page(frame, page_box)

            found = len(dots)
            missing = max(0, EXPECTED_DOTS - found)

            print(f"Pink dots found: {found}/{EXPECTED_DOTS}, missing: {missing}")

            debug = draw_debug(frame, page_box, dots)

            cv2.imshow("debug", debug)
            cv2.imshow("pink mask inside page", pink_mask)
            cv2.imshow("white page mask", page_mask)

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f}")

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