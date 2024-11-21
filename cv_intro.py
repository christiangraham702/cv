import cv2
import numpy as np

# Initialize video capture from the default camera
cap = cv2.VideoCapture(1)

# Add debug information about camera properties
print(f"Camera opened successfully: {cap.isOpened()}")
print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Add debug information about frame capture
    print(f"Frame captured successfully: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Display the original frame, grayscale, and edges
    cv2.imshow('Original', frame)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Edges', edges)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()