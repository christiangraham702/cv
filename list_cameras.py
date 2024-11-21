import cv2

def list_available_cameras():
    # Try indices from 0 to 10 (adjust range if needed)
    available_cameras = []
    
    for index in range(4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Try to get camera name (might not work on all systems)
            name = cap.getBackendName()
            
            # Read a test frame
            ret, frame = cap.read()
            if ret:
                print(f"\nCamera Index {index}:")
                print(f"Resolution: {width}x{height}")
                print(f"Backend: {name}")
                print(f"Frame shape: {frame.shape}")
                available_cameras.append(index)
            
            # Release the camera
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"\nAvailable camera indices: {available_cameras}")

if __name__ == "__main__":
    print("Scanning for available cameras...")
    list_available_cameras()