import cv2

# RTSP URL - Updated format for Tapo cameras
rtsp_url = "rtsp://camofted:tediscool@192.168.0.227:554/h264/ch1/main/av_stream"

# Open the RTSP stream with additional settings
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size

# Add connection verification
if not cap.isOpened():
    print(f"Failed to connect to stream: {rtsp_url}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame - attempting to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        continue
    # Display the frame
    cv2.imshow('Tapo C125 Feed', frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()