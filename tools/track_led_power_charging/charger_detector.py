import cv2
import numpy as np
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import configparser
import os
try:
    import winsound
except ImportError:
    winsound = None

class ChargerDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Default blue color range in RGB
        self.lower_blue = np.array([0, 0, 100])
        self.upper_blue = np.array([50, 50, 255])
        # Red color range in RGB
        self.lower_red = np.array([100, 0, 0])
        self.upper_red = np.array([255, 50, 50])
        
        # Charger body color range based on provided values (R=18, G=17, B=13)
        self.lower_body = np.array([max(0, 13-30), max(0, 17-30), max(0, 18-30)])
        self.upper_body = np.array([min(255, 13+30), min(255, 17+30), min(255, 18+30)])
        
        self.charging_start_time = None
        self.was_charging = False
        self.alert_shown = False
        self.current_frame = None
        self.pending_alert = None
        
        # Load calibration data from INI file
        self.load_calibration()
        
    def load_calibration(self):
        """
        Load calibration data from INI file
        """
        config = configparser.ConfigParser()
        if os.path.exists('calibration.ini'):
            config.read('calibration.ini')
            try:
                # Load blue calibration
                self.lower_blue = np.array([
                    config.getint('Blue', 'lower_r'),
                    config.getint('Blue', 'lower_g'),
                    config.getint('Blue', 'lower_b')
                ])
                self.upper_blue = np.array([
                    config.getint('Blue', 'upper_r'),
                    config.getint('Blue', 'upper_g'),
                    config.getint('Blue', 'upper_b')
                ])
                
                # Load red calibration
                self.lower_red = np.array([
                    config.getint('Red', 'lower_r'),
                    config.getint('Red', 'lower_g'),
                    config.getint('Red', 'lower_b')
                ])
                self.upper_red = np.array([
                    config.getint('Red', 'upper_r'),
                    config.getint('Red', 'upper_g'),
                    config.getint('Red', 'upper_b')
                ])
                
                # Load body calibration if exists
                if config.has_section('Body'):
                    self.lower_body = np.array([
                        config.getint('Body', 'lower_r'),
                        config.getint('Body', 'lower_g'),
                        config.getint('Body', 'lower_b')
                    ])
                    self.upper_body = np.array([
                        config.getint('Body', 'upper_r'),
                        config.getint('Body', 'upper_g'),
                        config.getint('Body', 'upper_b')
                    ])
                print("Calibration data loaded from calibration.ini")
            except Exception as e:
                print(f"Error loading calibration: {e}")
        
    def save_calibration(self):
        """
        Save calibration data to INI file
        """
        config = configparser.ConfigParser()
        
        # Save blue calibration
        config['Blue'] = {
            'lower_r': str(self.lower_blue[0]),
            'lower_g': str(self.lower_blue[1]),
            'lower_b': str(self.lower_blue[2]),
            'upper_r': str(self.upper_blue[0]),
            'upper_g': str(self.upper_blue[1]),
            'upper_b': str(self.upper_blue[2])
        }
        
        # Save red calibration
        config['Red'] = {
            'lower_r': str(self.lower_red[0]),
            'lower_g': str(self.lower_red[1]),
            'lower_b': str(self.lower_red[2]),
            'upper_r': str(self.upper_red[0]),
            'upper_g': str(self.upper_red[1]),
            'upper_b': str(self.upper_red[2])
        }
        
        # Save body calibration
        config['Body'] = {
            'lower_r': str(self.lower_body[0]),
            'lower_g': str(self.lower_body[1]),
            'lower_b': str(self.lower_body[2]),
            'upper_r': str(self.upper_body[0]),
            'upper_g': str(self.upper_body[1]),
            'upper_b': str(self.upper_body[2])
        }
        
        with open('calibration.ini', 'w') as configfile:
            config.write(configfile)
        print("Calibration data saved to calibration.ini")
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to update color space values when clicking on the frame
        Left click for red LED, right click for blue LED, middle click for body
        """
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            # Left click - record red LED color
            b, g, r = self.current_frame[y, x]
            
            # Update red color range using all three RGB values
            self.lower_red = np.array([max(0, r-50), max(0, g-50), max(0, b-50)])
            self.upper_red = np.array([min(255, r+50), min(255, g+50), min(255, b+50)])
            print(f"Red color range updated: R={r}, G={g}, B={b}")
            self.save_calibration()
                
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_frame is not None:
            # Right click - record blue LED color
            b, g, r = self.current_frame[y, x]
            
            # Update blue color range using all three RGB values
            self.lower_blue = np.array([max(0, r-50), max(0, g-50), max(0, b-50)])
            self.upper_blue = np.array([min(255, r+50), min(255, g+50), min(255, b+50)])
            print(f"Blue color range updated: R={r}, G={g}, B={b}")
            self.save_calibration()
            
        elif event == cv2.EVENT_MBUTTONDOWN and self.current_frame is not None:
            # Middle click - record body color
            b, g, r = self.current_frame[y, x]
            
            # Update body color range using all three RGB values
            self.lower_body = np.array([max(0, 0), max(0,0), max(0, 0)])
            self.upper_body = np.array([min(255, r+40), min(255, g+40), min(255, b+40)])
            print(f"Body color range updated: R={r}, G={g}, B={b}")
            self.save_calibration()
    
    def detect_charger_body(self, frame):
        """
        Detect the charger body region
        """
        # Create mask for charger body
        body_mask = cv2.inRange(frame, self.lower_body, self.upper_body)
        
        # Apply morphological operations to clean up the mask
        # Use ellipse kernel for better body detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assumed to be the charger body)
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Take the largest contour that meets minimum area requirement
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Increased minimum area threshold for body
                    return contour
        
        return None
    
    def detect_color_leds_in_region(self, frame, body_contour):
        """
wei111
        Detect LEDs within the body region
        """
        # Create a mask for the body region
        body_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if body_contour is not None:
            cv2.fillPoly(body_mask, [body_contour], 255)
        else:
            # If no body detected, use the whole frame
            body_mask.fill(255)

        # Apply the body mask to the frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=body_mask)

        # Convert to HSV for better color segmentation
        hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        
        # Split HSV channels
        h, s, v = cv2.split(hsv_frame)
        
        # Use brightness (V channel) to detect bright centers of LEDs
        # Lower threshold for high brightness (LED center)
        _, bright_mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)  # Reduced from 120
        
        # Apply morphological operations to clean up the bright mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the bright mask
        bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter bright contours by area to identify potential LED centers
        potential_leds = []
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            # Adjust area range to capture both small red LEDs and larger blue LEDs
            if 50 <= area <= 1000:  # Extended range to capture larger blue LEDs
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                potential_leds.append((center_x, center_y, w, h, area))
        
        # Now perform color detection around each potential LED
        blue_leds = []
        red_leds = []
        blue_filtered = []
        red_filtered = []

        # Process each potential LED
        for center_x, center_y, w, h, area in potential_leds:
            # Define a larger ROI around the center for color detection
            roi_size = max(w, h) * 3  # Increased from *2
            roi_x = max(0, center_x - roi_size // 2)
            roi_y = max(0, center_y - roi_size // 2)
            roi_x_end = min(masked_frame.shape[1], roi_x + roi_size)
            roi_y_end = min(masked_frame.shape[0], roi_y + roi_size)
            
            # Extract ROI
            roi = masked_frame[roi_y:roi_y_end, roi_x:roi_x_end]
            
            # Detect blue LEDs in ROI
            blue_mask_roi = cv2.inRange(roi, self.lower_blue, self.upper_blue)
            blue_mask_roi = cv2.morphologyEx(blue_mask_roi, cv2.MORPH_OPEN, kernel)
            blue_mask_roi = cv2.morphologyEx(blue_mask_roi, cv2.MORPH_CLOSE, kernel)
            
            # Detect red LEDs in ROI
            red_mask_roi = cv2.inRange(roi, self.lower_red, self.upper_red)
            red_mask_roi = cv2.morphologyEx(red_mask_roi, cv2.MORPH_OPEN, kernel)
            red_mask_roi = cv2.morphologyEx(red_mask_roi, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in blue mask
            blue_contours_roi, _ = cv2.findContours(blue_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in blue_contours_roi:
                contour_area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                    if 0.4 <= circularity <= 1.5:  # Slightly relaxed circularity for blue LEDs
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/float(h)
                        if 0.5 <= aspect_ratio <= 2.0:
                            # Calculate center relative to original frame
                            center_x_roi = x + w//2
                            center_y_roi = y + h//2
                            center_x_global = center_x_roi + roi_x
                            center_y_global = center_y_roi + roi_y
                            
                            # Check if this is likely a blue LED based on size
                            # Blue LEDs are typically larger than red LEDs
                            if contour_area > 100:  # Larger area threshold for blue LEDs
                                blue_leds.append((center_x_global, center_y_global, contour_area))
                                blue_filtered.append((contour, contour_area, True))
                            else:
                                # Might be a red LED or noise
                                red_filtered.append((contour, contour_area, False))
                        else:
                            blue_filtered.append((contour, contour_area, False))
                    else:
                        blue_filtered.append((contour, contour_area, False))
                else:
                    blue_filtered.append((contour, contour_area, False))
            
            # Find contours in red mask
            red_contours_roi, _ = cv2.findContours(red_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in red_contours_roi:
                contour_area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                    if 0.4 <= circularity <= 1.5:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w)/float(h)
                        if 0.5 <= aspect_ratio <= 2.0:
                            # Calculate center relative to original frame
                            center_x_roi = x + w//2
                            center_y_roi = y + h//2
                            center_x_global = center_x_roi + roi_x
                            center_y_global = center_y_roi + roi_y
                            
                            # Check if this is likely a red LED based on size
                            # Red LEDs are typically smaller than blue LEDs
                            if contour_area <= 100:  # Smaller area threshold for red LEDs
                                red_leds.append((center_x_global, center_y_global, contour_area))
                                red_filtered.append((contour, contour_area, True))
                            else:
                                # Might be a blue LED or noise
                                blue_filtered.append((contour, contour_area, False))
                        else:
                            red_filtered.append((contour, contour_area, False))
                    else:
                        red_filtered.append((contour, contour_area, False))
                else:
                    red_filtered.append((contour, contour_area, False))

        # Further refine LED detection by removing duplicates and resolving conflicts
        # Sort by area to prioritize larger LEDs as blue
        blue_leds = sorted(blue_leds, key=lambda x: x[2], reverse=True)
        red_leds = sorted(red_leds, key=lambda x: x[2])
        
        # Remove duplicates that are too close to each other (within 30 pixels)
        filtered_blue_leds = []
        for i, (bx, by, ba) in enumerate(blue_leds):
            is_duplicate = False
            for fx, fy, _ in filtered_blue_leds:
                distance = np.sqrt((bx - fx)**2 + (by - fy)**2)
                if distance < 30:  # If LEDs are too close, consider them duplicates
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_blue_leds.append((bx, by, ba))
                
        filtered_red_leds = []
        for i, (rx, ry, ra) in enumerate(red_leds):
            is_duplicate = False
            for fx, fy, _ in filtered_red_leds:
                distance = np.sqrt((rx - fx)**2 + (ry - fy)**2)
                if distance < 30:  # If LEDs are too close, consider them duplicates
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_red_leds.append((rx, ry, ra))
                
        # Remove red LEDs that are too close to blue LEDs (blue LEDs have priority)
        final_red_leds = []
        for rx, ry, ra in filtered_red_leds:
            too_close_to_blue = False
            for bx, by, _ in filtered_blue_leds:
                distance = np.sqrt((rx - bx)**2 + (ry - by)**2)
                if distance < 40:  # If red LED is too close to blue LED
                    too_close_to_blue = True
                    break
            if not too_close_to_blue:
                final_red_leds.append((rx, ry, ra))
                
        # Extract just the coordinates for compatibility with existing code
        final_blue_coords = [(x, y) for x, y, _ in filtered_blue_leds]
        final_red_coords = [(x, y) for x, y, _ in final_red_leds]
        
        return final_blue_coords, final_red_coords, blue_filtered, red_filtered
    
    def show_alert(self, duration):
        """
        Store alert information to be shown in the main thread
        """
        self.pending_alert = duration
    
    def display_alert(self):
        """
        Display the alert in the main thread
        """
        if self.pending_alert:
            # Play sound
            if winsound:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except:
                    pass  # In case of issues with sound
            
            # Create a simple visual alert on the frame
            print(f"ALERT: Battery charging completed! Total charging time: {self.pending_alert}")
            self.pending_alert = None
    
    def run(self):
        """
        Main loop for detecting charger status
        """
        cv2.namedWindow('Charger Status Detector')
        cv2.setMouseCallback('Charger Status Detector', self.mouse_callback)
        
        print("Left click on a red LED to calibrate red color detection")
        print("Right click on a blue LED to calibrate blue color detection")
        print("Middle click on the charger body to calibrate body detection")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            
            # Detect charger body
            body_contour = self.detect_charger_body(frame)
            
            # Detect LEDs within the body region
            blue_leds, red_leds, blue_filtered, red_filtered = self.detect_color_leds_in_region(frame, body_contour)
            
            # Draw all contours for debugging
            debug_frame = frame.copy()
            
            # Draw body contour
            if body_contour is not None:
                cv2.drawContours(debug_frame, [body_contour], -1, (0, 255, 0), 2)
            
            # Draw all contours for debugging
            for contour, area, *accepted in blue_filtered:
                if accepted and accepted[0]:
                    cv2.drawContours(debug_frame, [contour], -1, (255, 255, 0), 2)  # Cyan for accepted blue
                else:
                    cv2.drawContours(debug_frame, [contour], -1, (128, 128, 0), 1)  # Dark blue for rejected
            
            for contour, area, *accepted in red_filtered:
                if accepted and accepted[0]:
                    cv2.drawContours(debug_frame, [contour], -1, (0, 128, 255), 2)  # Orange for accepted red
                else:
                    cv2.drawContours(debug_frame, [contour], -1, (0, 0, 128), 1)  # Dark red for rejected
            
            # Draw detected LEDs
            for led in blue_leds:
                # print("LED:", led)
                cv2.circle(debug_frame, led, 15, (255, 0, 0), 2)  # Blue circle
            for led in red_leds:
                cv2.circle(debug_frame, led, 15, (0, 0, 255), 2)  # Red circle
            
            # Determine status based on LED colors
            status = ""
            # print("LEDs:", blue_leds, red_leds)
            if len(red_leds) >= 1 and len(blue_leds) >= 1:
                # One red and one blue LED = charging
                status = "Charging"  # "Charging" in Chinese
                if not self.was_charging:
                    self.charging_start_time = datetime.now()
                    self.was_charging = True
                    self.alert_shown = False
            elif len(blue_leds) >= 2 and len(red_leds) == 0:
                # Two blue LEDs = charging complete
                status = "Charging Complete"  # "Charging Complete" in Chinese
                
                # Show alert when charging completes
                if self.was_charging and not self.alert_shown:
                    if self.charging_start_time:
                        duration = datetime.now() - self.charging_start_time
                        minutes, seconds = divmod(duration.seconds, 60)
                        hours, minutes = divmod(minutes, 60)
                        
                        if hours > 0:
                            duration_str = f"{hours}h {minutes}m {seconds}s"
                        else:
                            duration_str = f"{minutes}m {seconds}s"
                        
                        # Show alert
                        self.show_alert(duration_str)
                        self.alert_shown = True
                self.was_charging = False
            else:
                # Handle intermediate states
                if len(blue_leds) >= 1 and len(red_leds) == 0:
                    status = "Waiting for charging"  # "Waiting for charging"
                else:
                    status = "Waiting for device..."  # "Waiting for device..."
                # Only reset charging state if we had a clear state before
                if len(blue_leds) == 0 and len(red_leds) == 0:
                    self.was_charging = False
            
            # Display alert if pending
            self.display_alert()
            
            # Display status on frame (top-left corner as required)
            cv2.putText(debug_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(debug_frame, "Left: red LED, Right: blue LED, Middle: body. Press 'q' to quit", (10, debug_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Charger Status Detector', debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ChargerDetector()
    detector.run()