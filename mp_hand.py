import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Finger indices
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

FINGER_PIPS = [
    mp_hands.HandLandmark.THUMB_IP,
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.PINKY_PIP,
]

# Open the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and check for extended fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the height and width of the frame
            h, w, _ = frame.shape

            for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
                # Get fingertip and PIP (proximal interphalangeal joint) coordinates
                fingertip = hand_landmarks.landmark[tip]
                finger_pip = hand_landmarks.landmark[pip]

                # Convert to pixel coordinates
                tip_x, tip_y = int(fingertip.x * w), int(fingertip.y * h)
                pip_x, pip_y = int(finger_pip.x * w), int(finger_pip.y * h)

                # Check if the finger is extended (fingertip above the PIP joint)
                if fingertip.y < finger_pip.y:  # Lower y value means higher in the frame
                    # Create a mapping of fingertip indices to finger names
                    finger_names = {
                        mp_hands.HandLandmark.THUMB_TIP: "Thumb",
                        mp_hands.HandLandmark.INDEX_FINGER_TIP: "Index",
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP: "Middle",
                        mp_hands.HandLandmark.RING_FINGER_TIP: "Ring",
                        mp_hands.HandLandmark.PINKY_TIP: "Pinky"
                    }
                    
                    cv2.circle(frame, (tip_x, tip_y), 10, (0, 255, 0), cv2.FILLED)
                    print(f"{finger_names[tip]} finger is extended")

    # Show the video feed with annotations
    cv2.imshow("Hand Tracking - Extended Fingers", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()