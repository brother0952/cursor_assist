import cv2
import numpy as np
import os


class OpenCVTemplate:
    """
    Base class for OpenCV templates
    """

    def __init__(self):
        self.cap = None
        self.window_name = "OpenCV Template"

    def list_available_cameras(self, max_cameras=10):
        """
        Check for available cameras up to max_cameras index
        
        Args:
            max_cameras (int): Maximum number of camera indices to check
            
        Returns:
            list: List of available camera indices
        """
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def select_camera(self):
        """
        Interactively select a camera from available ones
        
        Returns:
            int: Selected camera index
        """
        available = self.list_available_cameras()
        if not available:
            raise Exception("No cameras found")
            
        if len(available) == 1:
            print(f"Found 1 camera (index {available[0]}), using it automatically")
            return available[0]
            
        print("Available cameras:")
        for i, cam_idx in enumerate(available):
            print(f"  {i}: Camera {cam_idx}")
            
        while True:
            try:
                choice = int(input(f"Select camera (0-{len(available)-1}): "))
                if 0 <= choice < len(available):
                    return available[choice]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def initialize_camera(self, camera_index=None):
        """
        Initialize the camera. If camera_index is None, interactively select one.
        
        Args:
            camera_index (int, optional): Index of camera to open. If None, will prompt user to select.
        """
        if camera_index is None:
            camera_index = self.select_camera()
            
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {camera_index}")

    def release_resources(self):
        """Release camera and close windows"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Main execution method - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement the run method")


class Template1(OpenCVTemplate):
    """
    Template 1: Open camera, display frame in real-time,
    show RGB values at mouse click position directly on the OpenCV window
    """

    def __init__(self):
        super().__init__()
        self.mouse_x, self.mouse_y = 0, 0
        self.rgb_values = (0, 0, 0)
        self.window_name = "Template 1: Click to show RGB"

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x, self.mouse_y = x, y
            # Get the frame from the capture
            ret, frame = self.cap.read()
            if ret:
                # BGR to RGB - OpenCV uses BGR by default
                b, g, r = frame[self.mouse_y, self.mouse_x]
                self.rgb_values = (r, g, b)

    def run(self):
        """Run the template"""
        self.initialize_camera()
        
        # Set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Display RGB values on frame
            cv2.circle(frame, (self.mouse_x, self.mouse_y), 5, (0, 0, 255), -1)
            text = f"RGB: {self.rgb_values}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(self.window_name, frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()


class Template2(OpenCVTemplate):
    """
    Template 2: Open camera, display HSV values, add trackbar to adjust cv2.inRange parameters
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 2: HSV with Trackbars"
        self.hsv_window = "HSV Mask"
        
        # Trackbar initial values
        self.h_low, self.s_low, self.v_low = 0, 0, 0
        self.h_high, self.s_high, self.v_high = 179, 255, 255

    def nothing(self, x):
        """Dummy function for trackbars"""
        pass

    def create_trackbars(self):
        """Create trackbars for HSV range adjustment"""
        cv2.namedWindow(self.hsv_window)
        
        # Create trackbars for HSV low values
        cv2.createTrackbar('H Low', self.hsv_window, self.h_low, 179, self.nothing)
        cv2.createTrackbar('S Low', self.hsv_window, self.s_low, 255, self.nothing)
        cv2.createTrackbar('V Low', self.hsv_window, self.v_low, 255, self.nothing)
        
        # Create trackbars for HSV high values
        cv2.createTrackbar('H High', self.hsv_window, self.h_high, 179, self.nothing)
        cv2.createTrackbar('S High', self.hsv_window, self.s_high, 255, self.nothing)
        cv2.createTrackbar('V High', self.hsv_window, self.v_high, 255, self.nothing)

    def update_trackbar_values(self):
        """Update trackbar values"""
        self.h_low = cv2.getTrackbarPos('H Low', self.hsv_window)
        self.s_low = cv2.getTrackbarPos('S Low', self.hsv_window)
        self.v_low = cv2.getTrackbarPos('V Low', self.hsv_window)
        self.h_high = cv2.getTrackbarPos('H High', self.hsv_window)
        self.s_high = cv2.getTrackbarPos('S High', self.hsv_window)
        self.v_high = cv2.getTrackbarPos('V High', self.hsv_window)

    def run(self):
        """Run the template"""
        self.initialize_camera()
        self.create_trackbars()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get current trackbar positions
            self.update_trackbar_values()
            
            # Define range of colors in HSV
            lower_bound = np.array([self.h_low, self.s_low, self.v_low])
            upper_bound = np.array([self.h_high, self.s_high, self.v_high])
            
            # Threshold the HSV image to get only selected colors
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Display images
            cv2.imshow(self.window_name, frame)
            cv2.imshow(self.hsv_window, mask)
            cv2.imshow('Result', res)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()


class Template3(OpenCVTemplate):
    """
    Template 3: Similar to previous templates but with enhanced keyboard input recognition
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 3: Keyboard Controls"
        self.help_text = [
            "Keyboard Controls:",
            "'q' - Quit",
            "'s' - Save snapshot",
            "'g' - Toggle grayscale",
            "'f' - Toggle fullscreen"
        ]
        self.grayscale = False
        self.snapshot_counter = 0

    def display_help(self, frame):
        """Display help text on frame"""
        for i, text in enumerate(self.help_text):
            cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """Run the template with keyboard controls"""
        self.initialize_camera()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Apply grayscale if toggled
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back for display consistency

            # Display help text
            self.display_help(frame)

            # Display frame
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print("Exiting...")
                break
            elif key == ord('s'):  # Save snapshot
                filename = f"snapshot_{self.snapshot_counter}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                self.snapshot_counter += 1
            elif key == ord('g'):  # Toggle grayscale
                self.grayscale = not self.grayscale
            elif key == ord('f'):  # Toggle fullscreen
                # Note: Fullscreen toggle requires special handling depending on platform
                pass

        self.release_resources()


class Template4(OpenCVTemplate):
    """
    Template 4: Demonstrate common camera property adjustments
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 4: Camera Properties"
        self.properties = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS
        }
        self.property_names = list(self.properties.keys())
        self.current_prop_index = 0

    def display_properties(self, frame):
        """Display current camera properties on frame"""
        # Get current property values
        current_prop = self.property_names[self.current_prop_index]
        current_value = self.cap.get(self.properties[current_prop])
        
        # Display info
        info_lines = [
            f"Current Property: {current_prop}",
            f"Value: {current_value:.2f}",
            f"Use UP/DOWN arrows to change value",
            f"Use LEFT/RIGHT arrows to select property",
            f"Press 'q' to quit"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """Run the template with camera property adjustments"""
        self.initialize_camera()
        
        # Try setting some default properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("Available properties:")
        for i, prop in enumerate(self.property_names):
            print(f"{i}: {prop}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Display property info
            self.display_properties(frame)

            # Display frame
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(100) & 0xFF  # Slightly longer wait to make adjustments easier
            
            if key == ord('q'):  # Quit
                break
            elif key == 82:  # Up arrow - increase property value
                prop_id = self.properties[self.property_names[self.current_prop_index]]
                current_val = self.cap.get(prop_id)
                self.cap.set(prop_id, current_val + 1)
            elif key == 84:  # Down arrow - decrease property value
                prop_id = self.properties[self.property_names[self.current_prop_index]]
                current_val = self.cap.get(prop_id)
                self.cap.set(prop_id, current_val - 1)
            elif key == 83:  # Right arrow - next property
                self.current_prop_index = (self.current_prop_index + 1) % len(self.property_names)
            elif key == 81:  # Left arrow - previous property
                self.current_prop_index = (self.current_prop_index - 1) % len(self.property_names)

        self.release_resources()


class Template5(OpenCVTemplate):
    """
    Template 5: Find and display the largest contour in the frame
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 5: Largest Contour Detection"
        self.trackbar_window = "Controls"
        self.threshold_value = 127

    def nothing(self, x):
        """Dummy function for trackbars"""
        pass

    def create_control_trackbars(self):
        """Create trackbars for threshold control"""
        cv2.namedWindow(self.trackbar_window)
        cv2.createTrackbar('Threshold', self.trackbar_window, self.threshold_value, 255, self.nothing)

    def find_largest_contour(self, binary_image):
        """Find the largest contour in a binary image"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour
        else:
            return None

    def run(self):
        """Run the template to detect and display the largest contour"""
        self.initialize_camera()
        self.create_control_trackbars()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Get current threshold value
            self.threshold_value = cv2.getTrackbarPos('Threshold', self.trackbar_window)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
            
            # Find largest contour
            largest_contour = self.find_largest_contour(binary)
            
            # Draw the largest contour if it exists
            if largest_contour is not None:
                # Draw contour
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                
                # Calculate and display area
                area = cv2.contourArea(largest_contour)
                cv2.putText(frame, f"Largest Contour Area: {int(area)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Calculate and display centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Centroid: ({cx}, {cy})", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display images
            cv2.imshow(self.window_name, frame)
            cv2.imshow("Binary Image", binary)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()


class Template6(OpenCVTemplate):
    """
    Template 6: Calculate and display average brightness of the entire frame with real-time curve
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 6: Average Brightness"
        self.brightness_curve_window = "Brightness Curve"
        self.brightness_values = []  # Store brightness values over time
        self.max_values = 100  # Maximum number of values to display in the curve
        self.curve_image = np.zeros((200, 500, 3), dtype=np.uint8)  # Image for drawing the curve

    def calculate_average_brightness(self, frame):
        """
        Calculate the average brightness of a frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            float: Average brightness value (0-255)
        """
        # Convert BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        return avg_brightness

    def update_brightness_curve(self, brightness):
        """
        Update the brightness values list and draw the curve
        
        Args:
            brightness (float): Current brightness value
        """
        # Add new value
        self.brightness_values.append(brightness)
        
        # Keep only the last max_values entries
        if len(self.brightness_values) > self.max_values:
            self.brightness_values.pop(0)
        
        # Clear the curve image
        self.curve_image[:] = 0
        
        # Draw the curve
        if len(self.brightness_values) > 1:
            # Normalize values to fit in the curve image height (0-200)
            normalized_values = [(val / 255.0) * 200 for val in self.brightness_values]
            
            # Draw the curve line
            for i in range(1, len(normalized_values)):
                # Calculate coordinates
                x1 = int((i - 1) * (500 / self.max_values))
                y1 = int(200 - normalized_values[i - 1])  # Flip Y-axis (0 at top)
                x2 = int(i * (500 / self.max_values))
                y2 = int(200 - normalized_values[i])      # Flip Y-axis (0 at top)
                
                # Draw line segment
                cv2.line(self.curve_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw current value indicator
            current_y = int(200 - normalized_values[-1])
            cv2.circle(self.curve_image, (490, current_y), 5, (0, 0, 255), -1)
        
        # Draw reference lines
        # Middle line (127.5 brightness)
        cv2.line(self.curve_image, (0, 100), (500, 100), (128, 128, 128), 1)
        # Label reference lines
        cv2.putText(self.curve_image, "255", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        cv2.putText(self.curve_image, "127", (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        cv2.putText(self.curve_image, "0", (5, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    def run(self):
        """Run the template to display average brightness and its curve"""
        self.initialize_camera()
        
        # Create window for brightness curve
        cv2.namedWindow(self.brightness_curve_window)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Calculate average brightness
            avg_brightness = self.calculate_average_brightness(frame)
            
            # Update brightness curve
            self.update_brightness_curve(avg_brightness)
            
            # Display brightness on frame
            cv2.putText(frame, f"Avg Brightness: {avg_brightness:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow(self.window_name, frame)
            cv2.imshow(self.brightness_curve_window, self.curve_image)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()


class Template7(OpenCVTemplate):
    """
    Template 7: Save frame to save_pic directory when 's' key is pressed
    """

    def __init__(self):
        super().__init__()
        self.window_name = "Template 7: Press S to Save Picture"
        self.save_directory = "save_pic"
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"Created directory: {self.save_directory}")
        
        self.image_counter = 0

    def run(self):
        """Run the template"""
        self.initialize_camera()

        print("Instructions:")
        print("- Press 's' to save the current frame to save_pic directory")
        print("- Press 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            # Display instructions on frame
            cv2.putText(frame, "Press 'S' to save picture, 'Q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print("Exiting...")
                break
            elif key == ord('s'):  # Save image
                # Create filename with timestamp
                filename = os.path.join(self.save_directory, f"saved_img_{self.image_counter:04d}.jpg")
                success = cv2.imwrite(filename, frame)
                
                if success:
                    print(f"Image saved as: {filename}")
                    self.image_counter += 1
                else:
                    print(f"Failed to save image: {filename}")

        self.release_resources()


# Example usage:
# if __name__ == "__main__":
#     template = Template1()  # or Template2(), Template3(), Template4(), Template5(), Template6()
#     template.run()