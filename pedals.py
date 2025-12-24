import cv2
import mediapipe as mp

class FootGestureDetector:
    def __init__(self):
        # Initialize MediaPipe pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # To store the previous y-coordinate of the foot for gesture recognition
        self.prev_left_foot_y = None
        self.prev_right_foot_y = None
    
    def detect_gestures(self, frame):
        # Convert the frame to RGB as mediapipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get height of the frame
            height, width, _ = frame.shape
            
            # Extract foot landmarks (left and right)
            left_foot = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            
            # Convert normalized coordinates to pixel values
            left_foot_y = int(left_foot.y * height)
            right_foot_y = int(right_foot.y * height)
            
            # Gesture detection logic
            gesture = self._recognize_gesture(left_foot_y, right_foot_y)
            
            # Draw pose landmarks on the frame
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            return gesture, frame
        return None, frame

    def _recognize_gesture(self, left_foot_y, right_foot_y):
        gesture = None
        
        # Check for left foot movement (brake)
        if self.prev_left_foot_y is not None:
            if left_foot_y > self.prev_left_foot_y + 20:  # Foot moved down
                gesture = "Brake Pressed"
            elif left_foot_y < self.prev_left_foot_y - 20:  # Foot moved up
                gesture = "Brake Released"
        
        # Check for right foot movement (accelerator)
        if self.prev_right_foot_y is not None:
            if right_foot_y > self.prev_right_foot_y + 20:  # Foot moved down
                gesture = "Accelerator Pressed"
            elif right_foot_y < self.prev_right_foot_y - 20:  # Foot moved up
                gesture = "Accelerator Released"
        
        # Update previous y-coordinates
        self.prev_left_foot_y = left_foot_y
        self.prev_right_foot_y = right_foot_y
        
        return gesture
