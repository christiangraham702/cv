import cv2
import mediapipe as mp

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils 

# Setup video capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=6),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=5, circle_radius=8))

        # Here we can add logic to determine posture
        # This is a very basic approach and might need refinement
        # Example logic:
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y:
            posture = "Standing"
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y:
            posture = "Sitting"
        else:
            posture = "Lying Down"
        
        # Display posture on the frame
        cv2.putText(frame, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('MediaPipe Pose', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Clean up MediaPipe
pose.close()