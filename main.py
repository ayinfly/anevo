import subprocess
import time
import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

GREEN_LOWER = np.array([35, 60, 80])
GREEN_UPPER = np.array([90, 255, 255])

MIN_DOT_AREA = 20
MAX_DOT_AREA = 5000

DEBUG_EVERY_N_FRAMES = 15


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


def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def find_green_dot(frame):
    mask = green_mask(frame)

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

    if not dots:
        return None, mask, dots

    dots.sort(key=lambda p: p[2], reverse=True)
    return dots[0], mask, dots


def draw_debug(frame, dot, all_dots):
    debug = frame.copy()

    for i, (x, y, area) in enumerate(all_dots):
        cv2.circle(debug, (x, y), 15, (0, 255, 255), 2)
        cv2.putText(
            debug,
            f"{i}: {area:.0f}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if dot is not None:
        x, y, area = dot
        cv2.circle(debug, (x, y), 20, (0, 255, 0), 3)
        cv2.putText(
            debug,
            f"chosen ({x}, {y})",
            (x - 40, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite("debug_green_dot.jpg", debug)


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


def main():
    print("Starting fast rpicam green dot detector.")
    print("Press Ctrl+C to stop.")

    proc = start_camera()

    frame_count = 0
    start_time = time.time()

    try:
        for frame in mjpeg_frames(proc):
            frame_count += 1

            dot, mask, all_dots = find_green_dot(frame)

            if dot is None:
                print("No green dot found.")
            else:
                x, y, area = dot
                print(f"Green dot coords: x={x}, y={y}, area={area:.1f}")

            if frame_count % DEBUG_EVERY_N_FRAMES == 0:
                cv2.imwrite("debug_frame.jpg", frame)
                cv2.imwrite("debug_green_mask.jpg", mask)
                draw_debug(frame, dot, all_dots)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("Stopping.")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()