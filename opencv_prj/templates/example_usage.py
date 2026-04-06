"""
Example usage of OpenCV templates
"""

from opencv_templates import Template1, Template2, Template3, Template4, Template5, Template6, Template7

def main():
    print("OpenCV Templates Example")
    print("=" * 30)
    print("Select a template to run:")
    print("1. Template 1 - Mouse Click RGB Values")
    print("2. Template 2 - HSV Range Selection with Trackbars")
    print("3. Template 3 - Enhanced Keyboard Controls")
    print("4. Template 4 - Camera Property Adjustments")
    print("5. Template 5 - Largest Contour Detection")
    print("6. Template 6 - Average Brightness with Curve")
    print("7. Template 7 - Save Pictures with 'S' Key")
    print("0. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (0-7): "))
            
            if choice == 0:
                print("Goodbye!")
                break
            elif choice == 1:
                print("Running Template 1 - Mouse Click RGB Values")
                print("Click on the image to see RGB values. Press 'q' to quit.")
                template = Template1()
                template.run()
            elif choice == 2:
                print("Running Template 2 - HSV Range Selection with Trackbars")
                print("Adjust trackbars to modify color range. Press 'q' to quit.")
                template = Template2()
                template.run()
            elif choice == 3:
                print("Running Template 3 - Enhanced Keyboard Controls")
                print("Use 'q' to quit, 's' to save snapshot, 'g' for grayscale. Press 'q' to quit.")
                template = Template3()
                template.run()
            elif choice == 4:
                print("Running Template 4 - Camera Property Adjustments")
                print("Use arrow keys to adjust properties. Press 'q' to quit.")
                template = Template4()
                template.run()
            elif choice == 5:
                print("Running Template 5 - Largest Contour Detection")
                print("Adjust threshold with trackbar. Press 'q' to quit.")
                template = Template5()
                template.run()
            elif choice == 6:
                print("Running Template 6 - Average Brightness with Curve")
                print("Shows average brightness of frame and real-time curve. Press 'q' to quit.")
                template = Template6()
                template.run()
            elif choice == 7:
                print("Running Template 7 - Save Pictures with 'S' Key")
                print("Press 'S' to save the current frame to save_pic directory. Press 'q' to quit.")
                template = Template7()
                template.run()
            else:
                print("Invalid choice. Please enter a number between 0 and 7.")
                
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye!")
            break

def demo_multiple_cameras():
    """
    Demo function showing how to work with multiple cameras
    """
    print("Multiple Cameras Demo")
    print("=" * 20)
    
    # Create a template and let it auto-detect/select camera
    print("Auto-selecting camera:")
    template = Template1()
    template.run()
    
    # Or specify a camera index directly
    # template = Template1()
    # template.initialize_camera(0)  # Use camera 0 directly
    # template.run()

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Interactive template selection")
    print("2. Multiple cameras demo")
    
    try:
        mode = int(input("Enter mode (1 or 2): "))
        if mode == 1:
            main()
        elif mode == 2:
            demo_multiple_cameras()
        else:
            print("Invalid mode. Running interactive mode.")
            main()
    except ValueError:
        print("Invalid input. Running interactive mode.")
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")