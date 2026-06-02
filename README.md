# ANEVO

ANEVO is a paper-based virtual keyboard that lets a user type on a printed keyboard layout instead of a physical keyboard. The system uses a Raspberry Pi, camera input, computer vision, and keyboard-input forwarding to detect which printed key is being pressed and send that keypress to a connected computer. 

The main idea is to turn a normal sheet of paper into an interactive keyboard. The printed keyboard gives users a visible typing surface, while the software uses ArUco marker detection to identify which key regions are visible or covered. When a finger covers a marker, the system maps that marker to a key and forwards the input to the computer. This makes ANEVO a proof of concept for low-profile, space-saving, and adaptable computer peripherals.

[Link to our site!](https://sites.google.com/view/anevo/home)

## How It Works

The system is designed to use one camera which is angled and pointed at a printed keyboard. The printed keyboard uses ArUco markers, where each marker represents a key. First print out the attached aruco.pdf and place it in your desired typing location. Before running the program, the keyboard sheet should be placed clearly in view of the camera so the software can calibrate the marker positions.

Basic pipeline:

```text
Printed keyboard → Camera feed → ArUco detection → Calibration → Covered marker → Keypress
```

When a user covers a marker with their finger, the program treats that missing marker as a key press and sends the corresponding keyboard input using `evdev`.

## Repository Structure

```text
anevo/
├── Archive/                 # Older versions and experiments
├── aruco.pdf                # Printable keyboard layout
├── scan_aruco_ids.py        # Helper script for checking detected ArUco IDs
├── run_virtual_keyboard.py  # Main keyboard program
└── README.md
```

## Requirements

Hardware:

- Raspberry Pi
- Raspberry Pi camera
- Printed copy of `aruco.pdf`
- Good lighting

Software:

- Python 3
- `OpenCV`
- `NumPy`
- `evdev`
- `rpicam-vid`

On Raspberry Pi, install the main dependencies with:

```bash
sudo apt update
sudo apt install python3-opencv python3-evdev python3-numpy rpicam-apps
```

## Usage

Clone the repository:

```bash
git clone https://github.com/ayinfly/anevo.git
cd anevo
```

Print `aruco.pdf` and place it under the Raspberry Pi camera. Make sure the full page is visible.

To check whether the markers are detected correctly, run:

```bash
python3 scan_aruco_ids.py
```

To start the keyboard program, run:

```bash
sudo python3 run_virtual_keyboard.py
```

When the program starts, it will wait 30 seconds until all markers are visible, calibrate the keyboard, and then begin detecting key presses. This is done to gurantee your setup has good lighting. Begin typing as you would a normal keyboard.

Press `q` in the video window to quit.

## Supported Keys

The current version supports:

- A-Z letters
- Spacebar
- Enter
- Backspace
- Caps Lock

Caps Lock is handled inside the program as forcing all keys to `Shift + letter`.

## Notes

- The system works best with good lighting and a stable camera angle.
- The printed page should stay mostly still after calibration.
- The current version works best when pressing one key at a time.
- If detection is unreliable, run `scan_aruco_ids.py` to check which markers are visible.

## Team

- Abhijay Deevi
- Andrew Yin
- Eyaad Mir