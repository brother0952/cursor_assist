import cv2
import numpy as np

# Global variable to store the current frame
current_frame = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to print RGB values when clicking on the frame
    """
    global current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # Get BGR values (OpenCV uses BGR by default)
        b, g, r = current_frame[y, x]
        print(f"Position: ({x}, {y}) - RGB Values: R={r}, G={g}, B={b}")

def main():
    """
    Main function to capture video and print RGB values on click
    """
    global current_frame
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Debug RGB Values')
    cv2.setMouseCallback('Debug RGB Values', mouse_callback)
    
    print("Left click on the image to print RGB values at that position")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store the current frame for the mouse callback
        current_frame = frame.copy()
        
        # Display the frame
        cv2.imshow('Debug RGB Values', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()