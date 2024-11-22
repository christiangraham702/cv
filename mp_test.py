import cv2
import torch
import mediapipe as mp
import sys
from pathlib import Path

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import plot_one_box

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Run YOLOv5 object detection
    results = model(frame)
    detections = results.xyxy[0]  # Extract bounding boxes (x1, y1, x2, y2, confidence, class)

    # Filter for "person" class (class index 0 in COCO dataset)
    persons = [det for det in detections if int(det[-1]) == 0]

    # Step 2: Draw YOLOv5 bounding boxes
    for person in persons:
        x1, y1, x2, y2, conf, cls = person
        plot_one_box([x1, y1, x2, y2], frame, label=f"Person {conf:.2f}", color=(255, 0, 0), line_thickness=2)

        # Step 3: Crop and run pose estimation on detected person
        cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        if cropped_frame.size > 0:
            rgb_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_cropped)

            # If pose landmarks are detected
            if pose_results.pose_landmarks:
                # Map pose landmarks back to the original frame
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(x1 + landmark.x * (x2 - x1))
                    y = int(y1 + landmark.y * (y2 - y1))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Optionally: Draw pose landmarks and connections
                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the output frame
    cv2.imshow("Object Detection + Pose Estimation", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()