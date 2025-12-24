import cv2
import mediapipe as mp
import keyinput
from pedals import FootGestureDetector  

# Initialize MediaPipe Hands for steering control
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

font = cv2.FONT_HERSHEY_TRIPLEX

# Initialize cameras
cap_hand = cv2.VideoCapture(0)  # Camera for hand (steering control)
cap_foot = cv2.VideoCapture(2)  # Camera for foot (pedal control)

# Steering angle initialization (0% is straight, -100% full left, +100% full right)
steering_angle = 0
steering_increment = 10  # Percentage increment per hand movement

# To track the current state of brake and accelerator
brake_status = None
accelerator_status = None

# Gear shifting variables
gear = 4  # Start with gear 4 1aaawsws4dawd1
max_gear = 4  # Maximum gear
min_gear = 1  # Minimum gear

# Initialize FootGestureDetector for pedal control
detector = FootGestureDetector()

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


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
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
            xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
            radius = 150
            try:
                m = (co[1][1] - co[0][1]) / (co[1][0] - co[0][0])
            except ZeroDivisionError:
                m = float('inf')

            # Draw the steering wheel
            cv2.circle(image_hand, center=(int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=15)

            if co[0][0] > co[1][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65:
                # Turn left (increment steering angle by 10% until -100%)
                if steering_angle > -100:
                    steering_angle -= steering_increment
                keyinput.release_key('d')
                keyinput.press_key('a')
                cv2.putText(image_hand, f"Turn left ({steering_angle}%)", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            elif co[1][0] > co[0][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65:
                # Turn left (increment steering angle by 10% until -100%)
                if steering_angle > -100:
                    steering_angle -= steering_increment
                keyinput.release_key('d')
                keyinput.press_key('a')
                cv2.putText(image_hand, f"Turn left ({steering_angle}%)", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            elif co[0][0] > co[1][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65:
                # Turn right (increment steering angle by 10% until +100%)
                if steering_angle < 100:
                    steering_angle += steering_increment
                keyinput.release_key('a')
                keyinput.press_key('d')
                cv2.putText(image_hand, f"Turn right ({steering_angle}%)", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            elif co[1][0] > co[0][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65:
                # Turn right (increment steering angle by 10% until +100%)
                if steering_angle < 100:
                    steering_angle += steering_increment
                keyinput.release_key('a')
                keyinput.press_key('d')
                cv2.putText(image_hand, f"Turn right ({steering_angle}%)", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                # Go straight (release both 'a' and 'd' keys)
                steering_angle = 0
                keyinput.release_key('a')
                keyinput.release_key('d')
                cv2.putText(image_hand, "Straight", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Detect foot gestures for pedal control
        gesture, frame = detector.detect_gestures(image_foot)

        if gesture:
            if "Brake" in gesture:
                brake_status = gesture
                if "Pressed" in gesture:
                    keyinput.press_key("s")  # Press 'S' key
                else:
                    keyinput.release_key("s")  # Release 'S' key
            elif "Accelerator" in gesture:
                accelerator_status = gesture
                if "Pressed" in gesture:
                    keyinput.press_key("w")  # Press 'W' key
                else:
                    keyinput.release_key("w")  # Release 'W' key
        
        # Display brake status on the left side
        if brake_status:
            cv2.putText(frame, brake_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display accelerator status on the right side
        if accelerator_status:
            cv2.putText(frame, accelerator_status, (frame.shape[1] - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        

        # Display both images
        cv2.imshow('Hand Gesture Control (Steering and Gear)', image_hand)
        cv2.imshow('Foot Gesture Control (Pedals)', image_foot)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap_hand.release()
cap_foot.release()
cv2.destroyAllWindows()
