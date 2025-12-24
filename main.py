import cv2
import mediapipe as mp
import keyinput

# Initialize MediaPipe for hand and pose (foot) detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

font = cv2.FONT_HERSHEY_TRIPLEX

# Initialize cameras
cap_hand = cv2.VideoCapture(2)  # Camera for hand (steering control)
cap_foot = cv2.VideoCapture(0)  # Camera for foot (pedal control)

# Steering angle initialization (0% is straight, -100% full left, +100% full right)
steering_angle = 0
steering_increment = 10  # Percentage increment per hand movement

# To track the current state of brake and accelerator
brake_status = None
accelerator_status = None

# Gear shifting variables
gear = 4  # Start with gear 4
max_gear = 4  # Maximum gear
min_gear = 1  # Minimum gear

# Previous foot positions
prev_left_foot_y = None
prev_right_foot_y = None


def fingers_status(hand_landmarks):
    """Check the status of each finger (open or closed)."""
    finger_status = []

    # Thumb
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    finger_status.append(thumb_tip_y < thumb_mcp_y)

    # Index finger
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    finger_status.append(index_tip_y < index_mcp_y)

    # Middle finger
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    finger_status.append(middle_tip_y < middle_mcp_y)

    # Ring finger
    ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    finger_status.append(ring_tip_y < ring_mcp_y)

    # Pinky finger
    pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    finger_status.append(pinky_tip_y < pinky_mcp_y)

    return finger_status


def detect_gear(finger_status):
    """Detect the gear based on finger states."""
    if finger_status == [True, True, False, False, False]:
        return 1  # Parking (Gear 1)
    elif finger_status == [True, True, True, False, False]:
        return 2  # Gear 2
    elif finger_status == [True, True, True, True, False]:
        return 3  # Gear 3
    elif finger_status == [True, True, False, False, True]:
        return 4  # Gear 4
    else:
        return None  # No valid gear gesture detected


def detect_foot_gestures(frame, pose):
    global prev_left_foot_y, prev_right_foot_y

    # Convert the frame to RGB as mediapipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Get height of the frame
        height, width, _ = frame.shape
        
        # Extract foot landmarks (left and right)
        left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        
        # Convert normalized coordinates to pixel values
        left_foot_y = left_foot.y  # Normalized (0 to 1)
        right_foot_y = right_foot.y  # Normalized (0 to 1)
        
        # Gesture detection logic
        brake_status = detect_brake(left_foot_y)
        accelerator_status = detect_accelerator(right_foot_y)
        
        # Only draw foot landmarks for left and right foot
        cv2.circle(frame, (int(left_foot.x * width), int(left_foot.y * height)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_foot.x * width), int(right_foot.y * height)), 10, (0, 0, 255), -1)
        
        # Update previous positions
        prev_left_foot_y = left_foot_y
        prev_right_foot_y = right_foot_y
        
        return brake_status, accelerator_status, frame
    return None, None, frame


def detect_brake(left_foot_y):
    global prev_left_foot_y
    # Brake is pressed when the foot moves down (y-coordinate increases)
    if prev_left_foot_y is None:
        return None  # No previous data
    
    if left_foot_y > prev_left_foot_y:  # Foot moved down
        keyinput.press_key('s')  # Press the "S" key for brake
        return "Brake Pressed"
    elif left_foot_y < prev_left_foot_y:  # Foot moved up
        keyinput.release_key('s')  # Release the "S" key
        return "Brake Released"
    return None


def detect_accelerator(right_foot_y):
    global prev_right_foot_y
    # Accelerator is pressed when the foot moves down (y-coordinate increases)
    if prev_right_foot_y is None:
        return None  # No previous data
    
    if right_foot_y > prev_right_foot_y:  # Foot moved down
        keyinput.press_key('w')  # Press the "W" key for accelerator
        return "Accelerator Pressed"
    elif right_foot_y < prev_right_foot_y:  # Foot moved up
        keyinput.release_key('w')  # Release the "W" key
        return "Accelerator Released"
    return None


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, mp_pose.Pose(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7) as pose:
    
    while cap_hand.isOpened() and cap_foot.isOpened():
        # Capture hand gestures from the first camera
        success_hand, image_hand = cap_hand.read()
        if not success_hand:
            print("Ignoring empty hand camera frame.")
            continue

        # Capture foot gestures from the second camera
        success_foot, image_foot = cap_foot.read()
        if not success_foot:
            print("Ignoring empty foot camera frame.")
            continue

        # Prepare the image for hand processing
        image_hand.flags.writeable = False
        image_hand = cv2.cvtColor(image_hand, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image_hand)
        imageHeight, imageWidth, _ = image_hand.shape

        # Draw the hand annotations on the image
        image_hand.flags.writeable = True
        image_hand = cv2.cvtColor(image_hand, cv2.COLOR_RGB2BGR)
        co = []

        if results_hand.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hand.multi_hand_landmarks, results_hand.multi_handedness):
                mp_drawing.draw_landmarks(image_hand, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist coordinates
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_pixel_coords = (
                    int(wrist_landmark.x * imageWidth),
                    int(wrist_landmark.y * imageHeight)
                )
                co.append(wrist_pixel_coords)

                # Gear shifting logic for the right hand based on specific gestures
                if handedness.classification[0].label == 'Right':
                    finger_status = fingers_status(hand_landmarks)
                    detected_gear = detect_gear(finger_status)

                    if detected_gear and detected_gear != gear:
                        keyinput.release_key(str(gear))  # Release the previous gear key
                        gear = detected_gear  # Update the current gear
                        keyinput.press_key(str(gear))  # Press the new gear key

                    # Display the current gear
                    cv2.putText(image_hand, f"Gear: {gear}", (50, 100), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Steering logic based on both hands' positions
        if len(co) == 2:
            xm, ym = (co[0][0] + co[1][0]) // 2, (co[0][1] + co[1][1]) // 2
            dev = (imageWidth // 2 - xm) // steering_increment
            steering_angle = max(-100, min(100, steering_angle + dev))
            cv2.putText(image_hand, f"Steering Angle: {steering_angle}", (50, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # Steering control key presses
            if steering_angle > 0:
                keyinput.release_key('a')
                keyinput.press_key('d')
            elif steering_angle < 0:
                keyinput.release_key('d')
                keyinput.press_key('a')
            else:
                keyinput.release_key('a')
                keyinput.release_key('d')

        # Foot gestures processing
        brake_status, accelerator_status, image_foot = detect_foot_gestures(image_foot, pose)

        # Display brake and accelerator status on the foot camera feed
        if brake_status:
            cv2.putText(image_foot, brake_status, (50, 50), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        if accelerator_status:
            cv2.putText(image_foot, accelerator_status, (50, 100), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the combined images
        cv2.imshow('Hand Gestures (Steering)', image_hand)
        cv2.imshow('Foot Gestures (Pedals)', image_foot)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

cap_hand.release()
cap_foot.release()
cv2.destroyAllWindows()
