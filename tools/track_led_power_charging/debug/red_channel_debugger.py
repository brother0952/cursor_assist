import cv2
import numpy as np
import datetime

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get default resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Camera default resolution: {int(width)} x {int(height)}")
    print("Press 'S' to save current frame, 'Q' to quit")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Extract only the red channel
        filter_channel = frame.copy()
        keep_red_set1_keep_blue_set_0=0
        if keep_red_set1_keep_blue_set_0:
            filter_channel[:, :, 0] = 0  # Set blue channel to 0
            filter_channel[:, :, 1] = 0  # Set green channel to 0
            filter_channel[:, :, 2] = frame[:, :, 2]  # Keep red channel
        else :
            filter_channel[:, :, 0] = frame[:, :, 0]  # Keep blue channel
            filter_channel[:, :, 1] = 0  # Keep green channel
            filter_channel[:, :, 2] = 0  # Set red channel to 0
        
        # Display the red channel only
        cv2.imshow('Red Channel Debugger - Press S to save, Q to quit', filter_channel)
        
        # Wait for key input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'Q' to quit
        if key == ord('q') or key == ord('Q'):
            break
            
        # Press 'S' to save frame
        elif key == ord('s') or key == ord('S'):
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filter_channel_capture_{timestamp}.jpg"
            
            # Save image
            success = cv2.imwrite(filename, filter_channel)
            
            if success:
                print(f"Red channel frame saved as: {filename}")
            else:
                print("Error saving frame")

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()