import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define gesture detection
def detect_gesture(landmarks):
    # Extract needed landmarks
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    gesture = "NONE"

    # STOP: Right wrist above right shoulder
    if r_wrist.y < r_shoulder.y:
        gesture = "STOP ✋"

    # START: Both wrists above both shoulders
    if (r_wrist.y < r_shoulder.y) and (l_wrist.y < l_shoulder.y):
        gesture = "START 🚀"

    # MOVE LEFT: Left wrist extended left of left shoulder
    if l_wrist.x < l_shoulder.x - 0.1:  # threshold
        gesture = "MOVE LEFT ⬅️"

    # MOVE RIGHT: Right wrist extended right of right shoulder
    if r_wrist.x > r_shoulder.x + 0.1:  # threshold
        gesture = "MOVE RIGHT ➡️"

    # EMERGENCY: Wrists close together at chest level
    if abs(l_wrist.x - r_wrist.x) < 0.1 and abs(l_wrist.y - r_wrist.y) < 0.1:
        gesture = "EMERGENCY ❌"

    return gesture


# Start webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with Pose
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture = "NONE"

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            gesture = detect_gesture(landmarks)

        # Show detected gesture
        cv2.putText(image, f"Gesture: {gesture}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Display
        cv2.imshow("Industrial Gesture Control", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
