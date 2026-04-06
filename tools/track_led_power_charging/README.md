# Charger Status Detector

This program detects the charging status of a device by tracking the LED indicators on the charger.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- configparser (usually comes with Python)

## How It Works

The charger has two LED indicators:
1. First LED: Always displays blue
2. Second LED: 
   - Blue when no battery is connected or fully charged
   - Red when charging

Status detection:
- If one red and one blue LED are detected: "充电中" (Charging)
- If two blue LEDs are detected: "充电结束" (Charging Complete)

## Features

1. **Two-Stage Detection**: 
   - First detects the charger body to define the region of interest
   - Then searches for LEDs only within the charger body region
2. **Advanced Filtering**: Uses contour analysis to filter LED shapes, ignoring too large or too small detections
3. **Persistent Calibration**: 
   - Left-click on red LED to calibrate red color detection
   - Right-click on blue LED to calibrate blue color detection
   - Middle-click on charger body to calibrate body detection
   - Calibration data is saved to `calibration.ini` and loaded on startup
4. **Alert System**: When charging completes, shows notification with total charging time and plays a sound
5. **Real-time Monitoring**: Continuously monitors and displays the current status

## Usage

1. Connect a webcam or camera to your computer
2. Position the camera to capture the charger
3. Run the program:
   ```
   python charger_detector.py
   ```
4. If needed, calibrate detection:
   - Left-click on a red LED to calibrate red color range
   - Right-click on a blue LED to calibrate blue color range
   - Middle-click on the charger body to calibrate body detection
5. Press 'q' to quit the program

## Calibration

If the program doesn't correctly detect the charger or LEDs:
1. Left-click on a red LED to calibrate the red color range
2. Right-click on a blue LED to calibrate the blue color range
3. Middle-click on the charger body to calibrate the body detection

The program uses RGB color space for detection and applies morphological operations to clean up noise. All calibration data is automatically saved to `calibration.ini` and loaded when the program starts.

## Two-Stage Detection Process

1. **Body Detection**: 
   - Detects the main body of the charger to define the region of interest
   - Reduces false detections by ignoring areas outside the charger

2. **LED Detection**:
   - Only searches for LEDs within the identified charger body region
   - Applies contour filtering based on area, circularity, and aspect ratio

## Contour Filtering

To avoid false detections, the program filters contours by:
- Area (too small or too large detections are ignored)
- Circularity (contours that are not roughly circular are ignored)
- Aspect ratio (bounding rectangles that are not roughly square are ignored)

## Charging Completion Alert

When charging completes:
1. A message is printed to the console showing the total charging time
2. A sound is played (Windows only)
3. The alert only shows once per charging cycle

## Configuration File

Calibration data is automatically saved to and loaded from `calibration.ini`. This file contains the RGB values for red LED, blue LED, and charger body detection, ensuring consistent performance across sessions.

## Troubleshooting

- If LEDs are not detected, try adjusting lighting conditions or calibrating colors
- If false detections occur, ensure the camera is positioned correctly and calibrate all colors including the body
- Make sure the camera resolution is adequate to clearly see the charger and LEDs
- Ensure the LEDs appear roughly circular in the camera view