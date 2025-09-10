#!/usr/bin/env python3
"""
Simplified Hand Gesture Detection Module - MediaPipe-based Gesture Recognition
Detects 3 core gestures: Thumbs Up, Palm Stop, Arms Crossed (Emergency)
"""

import json
import time
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OpenCV and MediaPipe
try:
    import cv2
    import numpy as np
    import mediapipe as mp

    # Initialize MediaPipe solutions
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe and OpenCV loaded successfully")
except ImportError as e:
    logger.warning(f"MediaPipe/OpenCV not available: {e}. Using fallback mode.")
    MEDIAPIPE_AVAILABLE = False
    cv2 = None
    np = None
    mp = None


class SimplifiedSignDetector:
    """Simplified MediaPipe-based gesture detection for 3 core gestures"""

    def __init__(self):
        # Initialize MediaPipe models if available
        if MEDIAPIPE_AVAILABLE:
            try:
                # Initialize separate hand and pose models for better control
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                logger.info("SimplifiedSignDetector initialized with MediaPipe models")
            except Exception as e:
                logger.warning(f"MediaPipe initialization failed: {e}")
                self.hands = None
                self.pose = None
        else:
            self.hands = None
            self.pose = None
            logger.info("SimplifiedSignDetector initialized in fallback mode")

        # Detection parameters
        self.confidence_threshold = 0.75
        self.emergency_threshold = 0.8

        # Initialize logs
        self.detection_log = []
        self.emergency_log = []

        # Smoothing buffers for stable detection (shorter for responsiveness)
        self.gesture_buffer = deque(maxlen=3)
        self.emergency_buffer = deque(maxlen=4)
        
        # Supported gestures
        self.supported_gestures = {
            'THUMBS_UP': 'Thumb extended upward',
            'PALM_STOP': 'Open palm facing camera',
            'ARMS_CROSSED': 'Both arms crossed over chest (Emergency)'
        }

    def extract_landmarks(self, frame) -> Dict:
        """Extract hand and pose landmarks from frame"""
        try:
            if not MEDIAPIPE_AVAILABLE or not self.hands or not self.pose:
                return {'hands': [], 'pose': None}
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            landmarks_data = {
                'hands': [],
                'pose': None
            }
            
            # Process hands
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    hand_info = {
                        'landmarks': hand_landmarks,
                        'handedness': handedness.classification[0].label,
                        'confidence': handedness.classification[0].score
                    }
                    landmarks_data['hands'].append(hand_info)
            
            # Process pose
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                landmarks_data['pose'] = pose_results.pose_landmarks
            
            return landmarks_data
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return {'hands': [], 'pose': None}

    def detect_gesture(self, landmarks_data: Dict) -> Tuple[str, float, bool]:
        """Detect specific gesture from landmarks data"""
        try:
            hands = landmarks_data.get('hands', [])
            pose_landmarks = landmarks_data.get('pose', None)
            
            # Check for each of the 3 core gestures in priority order
            gestures_to_check = [
                ('ARMS_CROSSED', self.is_arms_crossed, hands, pose_landmarks),
                ('THUMBS_UP', self.is_thumbs_up, hands),
                ('PALM_STOP', self.is_palm_stop, hands),
            ]
            
            max_confidence = 0.0
            detected_gesture = "NONE"
            is_emergency = False
            
            for gesture_name, detection_func, *args in gestures_to_check:
                try:
                    confidence = detection_func(*args)
                    
                    if confidence > max_confidence and confidence > self.confidence_threshold:
                        max_confidence = confidence
                        detected_gesture = gesture_name
                        is_emergency = gesture_name == 'ARMS_CROSSED'
                        
                except Exception as e:
                    logger.warning(f"Error detecting {gesture_name}: {e}")
                    continue
            
            return detected_gesture, max_confidence, is_emergency
            
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return "NONE", 0.0, False

    def calculate_confidence(self, hands: List, pose_landmarks, additional_factor: float = 0.0) -> float:
        """Calculate confidence score based on landmark quality"""
        try:
            base_confidence = 0.75
            
            # Add hand confidence scores
            if hands:
                avg_hand_confidence = sum(hand['confidence'] for hand in hands) / len(hands)
                base_confidence = (base_confidence + avg_hand_confidence) / 2
            
            # Add pose confidence if available
            if pose_landmarks:
                base_confidence += 0.05
            
            # Add gesture-specific confidence
            base_confidence += additional_factor
            
            return min(base_confidence, 1.0)
            
        except Exception:
            return 0.75

    def is_thumbs_up(self, hands) -> float:
        """Detect thumbs up gesture with high confidence"""
        try:
            if not hands:
                return 0.0
                
            for hand_info in hands:
                hand_landmarks = hand_info['landmarks']
                
                # Get key landmark positions
                thumb_tip = hand_landmarks.landmark[4]
                thumb_ip = hand_landmarks.landmark[3]
                thumb_mcp = hand_landmarks.landmark[2]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                wrist = hand_landmarks.landmark[0]
                
                # Check if thumb is extended upward
                thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
                thumb_extended = abs(thumb_tip.y - wrist.y) > 0.1
                
                # Check if other fingers are curled (below thumb level)
                other_fingers_down = (
                    index_tip.y > thumb_tip.y + 0.05 and
                    middle_tip.y > thumb_tip.y + 0.05 and
                    ring_tip.y > thumb_tip.y + 0.05 and
                    pinky_tip.y > thumb_tip.y + 0.05
                )
                
                if thumb_up and thumb_extended and other_fingers_down:
                    confidence_boost = 0.15 if hand_info['confidence'] > 0.8 else 0.1
                    return self.calculate_confidence(hands, None, confidence_boost)
                    
            return 0.0
        except Exception as e:
            logger.error(f"Error in thumbs_up detection: {e}")
            return 0.0

    def is_palm_stop(self, hands) -> float:
        """Detect open palm stop gesture"""
        try:
            if not hands:
                return 0.0
                
            for hand_info in hands:
                hand_landmarks = hand_info['landmarks']
                
                # Get fingertip and joint positions
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                index_pip = hand_landmarks.landmark[6]
                middle_tip = hand_landmarks.landmark[12]
                middle_pip = hand_landmarks.landmark[10]
                ring_tip = hand_landmarks.landmark[16]
                ring_pip = hand_landmarks.landmark[14]
                pinky_tip = hand_landmarks.landmark[20]
                pinky_pip = hand_landmarks.landmark[18]
                wrist = hand_landmarks.landmark[0]
                
                # Check if fingers are extended (tips higher than joints)
                fingers_extended = (
                    index_tip.y < index_pip.y and
                    middle_tip.y < middle_pip.y and
                    ring_tip.y < ring_pip.y and
                    pinky_tip.y < pinky_pip.y
                )
                
                # Check if thumb is extended outward
                thumb_extended = abs(thumb_tip.x - wrist.x) > 0.08
                
                # Check if palm is relatively vertical (stop gesture)
                palm_vertical = all([
                    tip.y < wrist.y - 0.1 for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
                ])
                
                if fingers_extended and thumb_extended and palm_vertical:
                    confidence_boost = 0.12 if hand_info['confidence'] > 0.8 else 0.08
                    return self.calculate_confidence(hands, None, confidence_boost)
                    
            return 0.0
        except Exception as e:
            logger.error(f"Error in palm_stop detection: {e}")
            return 0.0

    def is_arms_crossed(self, hands, pose_landmarks) -> float:
        """Detect arms crossed gesture (Emergency signal)"""
        try:
            if len(hands) < 2 or not pose_landmarks:
                return 0.0
            
            # Get pose landmarks
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate arm crossing
            # Check if left wrist is towards right side and vice versa
            left_arm_crossed = left_wrist.x > (left_shoulder.x + right_shoulder.x) / 2
            right_arm_crossed = right_wrist.x < (left_shoulder.x + right_shoulder.x) / 2
            
            # Check if arms are at appropriate height (chest level)
            chest_center_y = (left_shoulder.y + right_shoulder.y) / 2
            left_arm_chest_level = abs(left_wrist.y - chest_center_y) < 0.15
            right_arm_chest_level = abs(right_wrist.y - chest_center_y) < 0.15
            
            # Check if elbows are bent (not arms straight out)
            left_elbow_bent = left_elbow.y > chest_center_y - 0.1
            right_elbow_bent = right_elbow.y > chest_center_y - 0.1
            
            # All conditions must be met for arms crossed
            if (left_arm_crossed and right_arm_crossed and 
                left_arm_chest_level and right_arm_chest_level and
                left_elbow_bent and right_elbow_bent):
                
                # High confidence for emergency gesture
                confidence_boost = 0.2
                return self.calculate_confidence(hands, pose_landmarks, confidence_boost)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in arms_crossed detection: {e}")
            return 0.0

    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        try:
            # Extract landmarks
            landmarks_data = self.extract_landmarks(frame)
            
            # Detect gesture
            gesture, confidence, is_emergency = self.detect_gesture(landmarks_data)
            
            # Add to buffer for smoothing
            self.gesture_buffer.append((gesture, confidence, is_emergency))
            
            # Smooth gesture detection
            if gesture != "NONE" and confidence > self.confidence_threshold:
                # Check if gesture is consistent
                recent_gestures = [g[0] for g in list(self.gesture_buffer)[-2:]]
                if all(g == gesture for g in recent_gestures):
                    smoothed_gesture = gesture
                    smoothed_confidence = confidence
                    smoothed_emergency = is_emergency
                else:
                    smoothed_gesture = "NONE"
                    smoothed_confidence = 0.0
                    smoothed_emergency = False
            else:
                smoothed_gesture = "NONE"
                smoothed_confidence = 0.0
                smoothed_emergency = False
            
            # Create result
            result = {
                'gesture': smoothed_gesture,
                'confidence': smoothed_confidence,
                'is_emergency': smoothed_emergency,
                'hands_detected': len(landmarks_data.get('hands', [])),
                'landmarks': landmarks_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log detection
            if smoothed_gesture != "NONE":
                self.detection_log.append(result.copy())
                if len(self.detection_log) > 100:
                    self.detection_log = self.detection_log[-100:]
                
                # Log emergency
                if smoothed_emergency:
                    self.emergency_log.append(result.copy())
                    if len(self.emergency_log) > 50:
                        self.emergency_log = self.emergency_log[-50:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'gesture': 'ERROR',
                'confidence': 0.0,
                'is_emergency': False,
                'hands_detected': 0,
                'landmarks': {'hands': [], 'pose': None},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_detection_logs(self, limit: int = 100):
        """Get recent detection logs"""
        return self.detection_log[-limit:]

    def get_emergency_logs(self, limit: int = 50):
        """Get recent emergency logs"""
        return self.emergency_log[-limit:]

    def get_statistics(self):
        """Get detection statistics"""
        try:
            total_detections = len(self.detection_log)
            emergency_detections = len(self.emergency_log)
            
            if total_detections == 0:
                return {
                    'total_detections': 0,
                    'emergency_rate': 0.0,
                    'most_common_gesture': 'NONE',
                    'avg_confidence': 0.0
                }
            
            # Calculate statistics
            emergency_rate = (emergency_detections / total_detections) * 100
            
            # Most common gesture
            gesture_counts = {}
            total_confidence = 0.0
            for log in self.detection_log:
                gesture = log['gesture']
                confidence = log['confidence']
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                total_confidence += confidence
            
            most_common_gesture = max(gesture_counts, key=gesture_counts.get) if gesture_counts else 'NONE'
            avg_confidence = total_confidence / total_detections
            
            return {
                'total_detections': total_detections,
                'emergency_rate': emergency_rate,
                'most_common_gesture': most_common_gesture,
                'avg_confidence': avg_confidence,
                'gesture_counts': gesture_counts
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'total_detections': 0,
                'emergency_rate': 0.0,
                'most_common_gesture': 'NONE',
                'avg_confidence': 0.0
            }

    def clear_logs(self):
        """Clear all detection logs"""
        self.detection_log.clear()
        self.emergency_log.clear()


# Global variables
_sign_detector = None


def initialize_sign_detector():
    """Initialize the global simplified sign detector"""
    global _sign_detector
    try:
        logger.info("Initializing simplified sign detector...")
        _sign_detector = SimplifiedSignDetector()
        logger.info("Simplified sign detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize sign detector: {e}")
        _sign_detector = None
        return False


def get_sign_detector():
    """Get the global sign detector instance"""
    return _sign_detector


if __name__ == "__main__":
    # Test the detector
    if initialize_sign_detector():
        detector = get_sign_detector()
        print("Simplified Sign Detector initialized successfully!")
        print("Supported gestures:", detector.supported_gestures)
    else:
        print("Failed to initialize sign detector")