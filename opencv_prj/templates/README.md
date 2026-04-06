# OpenCV Templates

A collection of reusable OpenCV templates for common computer vision tasks. Each template is implemented as a separate class for easy reference and extension.

## Prerequisites

Install the required packages:

```bash
pip install opencv-python numpy
```

## Templates Overview

### Template 1: Mouse Click RGB Values
Opens the camera and displays live video feed. When you click on the image, it shows the RGB values at that point directly on the OpenCV window.

### Template 2: HSV Range Selection with Trackbars
Displays the camera feed in HSV color space with adjustable trackbars to fine-tune the cv2.inRange parameters for color filtering.

### Template 3: Enhanced Keyboard Controls
Similar to previous templates but with enhanced keyboard input handling including:
- 'q' to quit
- 's' to save snapshots
- 'g' to toggle grayscale
- 'f' to toggle fullscreen

### Template 4: Camera Property Adjustments
Demonstrates how to adjust common camera properties such as:
- Brightness
- Contrast
- Saturation
- Hue
- Gain
- Exposure
- Resolution (width/height)
- Frame rate

Use arrow keys to navigate and adjust properties.

### Template 5: Largest Contour Detection
Finds and highlights the largest contour in the frame with additional information:
- Contour area
- Bounding rectangle
- Centroid

Includes a trackbar to adjust the threshold value for binary conversion.

### Template 6: Average Brightness with Real-time Curve
Calculates and displays the average brightness of the entire frame with a real-time curve showing brightness changes over time. The main window shows the current average brightness value, while a secondary window displays the brightness curve.

### Template 7: Save Pictures with 'S' Key
Allows capturing and saving frames from the camera feed to a dedicated directory. When running this template:
- Press 'S' to save the current frame to the `save_pic` directory
- Press 'Q' to quit the application
- Saved images are automatically numbered sequentially (saved_img_0000.jpg, saved_img_0001.jpg, etc.)

## Multi-Camera Support

The base [OpenCVTemplate](file://d:\git\py_process\opencv_prj\templates\opencv_templates.py#L5-L5) class now includes automatic detection of available cameras. When initializing a template without specifying a camera index, the system will:

1. Scan for available cameras (indices 0-9)
2. If multiple cameras are found, prompt the user to select one
3. If only one camera is found, use it automatically

You can still specify a camera index directly if desired:

```python
template = Template1()
template.initialize_camera(1)  # Use camera with index 1
template.run()
```

## Usage

To use any template, create an instance of the corresponding class and call its `run()` method:

```python
from opencv_templates import Template1, Template2, Template3, Template4, Template5, Template6, Template7

# Example for Template 1 with automatic camera selection
template = Template1()
template.run()

# Example for Template 1 with specific camera index
template = Template1()
template.initialize_camera(0)  # Use camera 0
template.run()

# Example for Template 7 - saving pictures with 'S' key
template = Template7()
template.run()
```

Press 'q' to quit any of the templates.

## Extending Templates

All templates inherit from the `OpenCVTemplate` base class, making it easy to extend with custom functionality:

```python
from opencv_templates import OpenCVTemplate

class MyCustomTemplate(OpenCVTemplate):
    def run(self):
        self.initialize_camera()
        # Your custom implementation here
        self.release_resources()
```