#!/usr/bin/env python3
"""
Hand Gesture Detection Module - MediaPipe-based Gesture Recognition
Detects specific gestures: Help Signal, Stop Signal, OK Sign, Thumbs Up/Down,
Call Gesture, Chest Tap, Arms Crossed (Emergency), Cross Arms Above Head
"""

import json
import time
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import deque, defaultdict

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
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe hands, pose, and OpenCV loaded successfully")
except ImportError as e:
    logger.warning(f"MediaPipe/OpenCV not available: {e}. Using fallback mode.")
    MEDIAPIPE_AVAILABLE = False
    cv2 = None
    np = None
    mp = None

# MediaPipe processor for extracting hand and pose landmarks
def extract_landmarks(frame):
    """Extract hand and pose landmarks from frame"""
    results = {}
    with mp_hands.Hands(static_image_mode=True) as hands:
        hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            results['hands'] = [lm for lm in hand_results.multi_hand_landmarks]
        else:
            results['hands'] = []
    with mp_pose.Pose(static_image_mode=True) as pose:
        pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pose_results.pose_landmarks:
            results['pose'] = pose_results.pose_landmarks
        else:
            results['pose'] = None
    return results

class SignDetector:
    """MediaPipe-based gesture detection system for specific gestures"""

    def __init__(self):
        # Initialize MediaPipe models if available
        if MEDIAPIPE_AVAILABLE:
            try:
                # Initialize holistic model for both hands and pose
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("SignDetector initialized with MediaPipe holistic model")
            except Exception as e:
                logger.warning(f"MediaPipe initialization failed: {e}")
                self.holistic = None
        else:
            self.holistic = None
            logger.info("SignDetector initialized in fallback mode")

        # Detection parameters
        self.confidence_threshold = 0.7
        self.emergency_threshold = 0.8

        # Initialize logs
        self.detection_log = []
        self.emergency_log = []

        # Smoothing buffers for stable detection
        self.gesture_buffer = deque(maxlen=5)
        self.emergency_buffer = deque(maxlen=6)
    
    def extract_landmarks(self, frame) -> Dict:
        """Extract hand and pose landmarks from frame"""
        try:
            if not MEDIAPIPE_AVAILABLE or not self.holistic:
                return {'hands': [], 'pose': None, 'frame_shape': (480, 640, 3)}
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_data = {
                'hands': [],
                'pose': None,
                'frame_shape': frame.shape
            }
            
            # Process holistic landmarks (hands + pose)
            results = self.holistic.process(rgb_frame)
            
            # Extract hand landmarks
            if results.left_hand_landmarks:
                hand_info = {
                    'landmarks': results.left_hand_landmarks,
                    'handedness': 'Left',
                    'confidence': 0.8  # MediaPipe doesn't provide confidence for holistic hands
                }
                results_data['hands'].append(hand_info)
                
            if results.right_hand_landmarks:
                hand_info = {
                    'landmarks': results.right_hand_landmarks,
                    'handedness': 'Right',
                    'confidence': 0.8
                }
                results_data['hands'].append(hand_info)
            
            # Extract pose landmarks
            if results.pose_landmarks:
                results_data['pose'] = results.pose_landmarks
            
            return results_data
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return {'hands': [], 'pose': None, 'frame_shape': frame.shape}
    
    def detect_gesture(self, landmarks_data: Dict) -> Tuple[str, float, bool]:
        """Detect specific gestures using MediaPipe landmarks"""
        try:
            hands = landmarks_data.get('hands', [])
            pose = landmarks_data.get('pose', None)
            
            # Single hand gestures - check each hand separately
            for hand_info in hands:
                hand_landmarks = hand_info['landmarks']
                handedness = hand_info['handedness']
                
                # Help Signal - Open hand with fingers extended
                if self.is_help_signal(hand_landmarks):
                    return 'HELP_SIGNAL', 0.85, False
                
                # Stop Signal - Open palm facing forward
                if self.is_stop_signal(hand_landmarks):
                    return 'STOP_SIGNAL', 0.85, False
                
                # OK Sign - Circle with thumb and index
                if self.is_ok_sign(hand_landmarks):
                    return 'OK_SIGN', 0.90, False
                
                # Thumbs Up
                if self.is_thumbs_up(hand_landmarks):
                    return 'THUMBS_UP', 0.90, False
                
                # Thumbs Down
                if self.is_thumbs_down(hand_landmarks):
                    return 'THUMBS_DOWN', 0.90, False
                
                # Call Gesture - Phone to ear
                if self.is_call_gesture(hand_landmarks):
                    return 'CALL_GESTURE', 0.85, False
            
            # Two-hand and pose-based gestures
            if pose and hands:
                # Arms Crossed - EMERGENCY GESTURE
                if self.is_arms_crossed(hands, pose):
                    return 'ARMS_CROSSED', 0.95, True
                
                # Cross Arms Above Head
                if self.is_cross_arms_above_head(hands, pose):
                    return 'CROSS_ARMS_ABOVE_HEAD', 0.85, False
                
                # Chest Tap
                if self.is_chest_tap(hands, pose):
                    return 'CHEST_TAP', 0.80, False
            
            return 'NONE', 0.0, False
            
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return 'ERROR', 0.0, False
    
    # Utility functions
    def dist(self, a, b):
        """Calculate distance between two points"""
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def angle(self, a, b, c):
        """Calculate angle between three points"""
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        dot = ab[0] * cb[0] + ab[1] * cb[1]
        mag = math.hypot(ab[0], ab[1]) * math.hypot(cb[0], cb[1]) + 1e-6
        cosv = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(cosv))
    
    # Gesture detection functions - Enhanced with comprehensive rule-based detection
    def is_help_signal(self, hand_landmarks):
        """Help Signal - Open hand with all fingers extended"""
        try:
            # Check if fingers are extended
            fingers = []

            # Thumb (different logic due to orientation)
            if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:  # Right hand
                fingers.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
            else:  # Left hand
                fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)

            # Other fingers
            for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

            return sum(fingers) >= 4
        except:
            return False

    def is_stop_signal(self, hand_landmarks):
        """Stop Signal - Open palm facing forward"""
        # Same as help signal - open hand
        return self.is_help_signal(hand_landmarks)

    def is_ok_sign(self, hand_landmarks):
        """OK Sign - Thumb and index finger form a circle"""
        try:
            thumb_x = hand_landmarks.landmark[4].x
            thumb_y = hand_landmarks.landmark[4].y
            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y
            distance = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
            return distance < 0.2
        except:
            return False

    def is_thumbs_up(self, hand_landmarks):
        """Thumbs Up - Thumb extended upward, other fingers curled"""
        try:
            # Thumb tip above index tip (with margin for tilt)
            thumb_y = hand_landmarks.landmark[4].y
            index_y = hand_landmarks.landmark[8].y
            wrist_y = hand_landmarks.landmark[0].y
            # Also check thumb is above wrist and hand is upright
            return thumb_y < index_y + 0.15 and thumb_y < wrist_y
        except:
            return False

    def is_thumbs_down(self, hand_landmarks):
        """Thumbs Down - Thumb extended downward"""
        try:
            # Thumb tip below index tip
            thumb_y = hand_landmarks.landmark[4].y
            index_y = hand_landmarks.landmark[8].y
            return thumb_y > index_y
        except:
            return False

    def is_call_gesture(self, hand_landmarks):
        """Call Gesture - Phone sign: thumb near ear, pinky near mouth"""
        try:
            thumb = hand_landmarks.landmark[4]
            pinky = hand_landmarks.landmark[20]
            # Approximate positions, account for mirroring
            return ((thumb.x < 0.3 or thumb.x > 0.7) and (pinky.x > 0.7 or pinky.x < 0.3))
        except:
            return False

    def is_fist(self, hand_landmarks):
        """Fist - All fingers curled"""
        try:
            # All fingers curled: tips below knuckles, allow 3 or more curled
            thumb_tip = hand_landmarks.landmark[4].y
            thumb_knuckle = hand_landmarks.landmark[2].y
            index_tip = hand_landmarks.landmark[8].y
            index_knuckle = hand_landmarks.landmark[6].y
            middle_tip = hand_landmarks.landmark[12].y
            middle_knuckle = hand_landmarks.landmark[10].y
            ring_tip = hand_landmarks.landmark[16].y
            ring_knuckle = hand_landmarks.landmark[14].y
            pinky_tip = hand_landmarks.landmark[20].y
            pinky_knuckle = hand_landmarks.landmark[18].y
            curled_count = sum([
                thumb_tip > thumb_knuckle,
                index_tip > index_knuckle,
                middle_tip > middle_knuckle,
                ring_tip > ring_knuckle,
                pinky_tip > pinky_knuckle
            ])
            return curled_count >= 3
        except:
            return False

    def is_peace(self, hand_landmarks):
        """Peace Sign - Index and middle extended, others curled"""
        try:
            # Index and middle extended, others curled, allow 2 extended
            index_tip = hand_landmarks.landmark[8].y
            index_knuckle = hand_landmarks.landmark[6].y
            middle_tip = hand_landmarks.landmark[12].y
            middle_knuckle = hand_landmarks.landmark[10].y
            ring_tip = hand_landmarks.landmark[16].y
            ring_knuckle = hand_landmarks.landmark[14].y
            pinky_tip = hand_landmarks.landmark[20].y
            pinky_knuckle = hand_landmarks.landmark[18].y
            extended_count = sum([
                index_tip < index_knuckle,
                middle_tip < middle_knuckle
            ])
            curled_count = sum([
                ring_tip > ring_knuckle,
                pinky_tip > pinky_knuckle
            ])
            return extended_count == 2 and curled_count >= 1
        except:
            return False

    def is_cover_mouth(self, hand_landmarks):
        """Cover Mouth - Hand over mouth"""
        try:
            # Hand over mouth: fingers near center
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            return (0.4 < thumb.x < 0.6 and 0.4 < index.x < 0.6 and 0.4 < middle.x < 0.6)
        except:
            return False

    def is_touch_ear(self, hand_landmarks):
        """Touch Ear - Thumb near ear"""
        try:
            thumb = hand_landmarks.landmark[4]
            pinky = hand_landmarks.landmark[20]
            return ((thumb.x - pinky.x)**2 + (thumb.y - pinky.y)**2)**0.5 < 0.1
        except:
            return False

    def is_point_down(self, hand_landmarks):
        """Pointing Down - Index extended down"""
        try:
            index_tip = hand_landmarks.landmark[8].y
            index_knuckle = hand_landmarks.landmark[6].y
            return index_tip > index_knuckle
        except:
            return False

    def is_point_up(self, hand_landmarks):
        """Pointing Up - Index extended up"""
        try:
            index_tip = hand_landmarks.landmark[8].y
            index_knuckle = hand_landmarks.landmark[6].y
            return index_tip < index_knuckle
        except:
            return False

    def is_point_left(self, hand_landmarks):
        """Pointing Left - Index to left"""
        try:
            index_tip = hand_landmarks.landmark[8].x
            wrist = hand_landmarks.landmark[0].x
            return index_tip < wrist
        except:
            return False

    def is_point_right(self, hand_landmarks):
        """Pointing Right - Index to right"""
        try:
            index_tip = hand_landmarks.landmark[8].x
            wrist = hand_landmarks.landmark[0].x
            return index_tip > wrist
        except:
            return False

    def is_palm_up(self, hand_landmarks):
        """Palm Up - Beckoning"""
        try:
            palm = hand_landmarks.landmark[9]  # Middle finger MCP
            wrist = hand_landmarks.landmark[0]
            return palm.y < wrist.y
        except:
            return False

    def is_wave_one_hand(self, hand_landmarks):
        """Wave One Hand - Open hand"""
        return self.is_help_signal(hand_landmarks)

    # Pose-based gesture detection functions
    def is_stop_pose(self, pose_landmarks):
        """Stop Pose - Both arms raised, palms forward"""
        try:
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            # Wrists above shoulders
            return (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y)
        except:
            return False

    def is_arms_crossed(self, hands, pose_landmarks):
        """Arms Crossed - Emergency gesture"""
        try:
            left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            # Elbows close to each other
            return abs(left_elbow.x - right_elbow.x) < 0.1
        except:
            return False

    def is_hands_up(self, pose_landmarks):
        """Hands Up - Hands raised above head"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return (left_wrist.y < nose.y and right_wrist.y < nose.y)
        except:
            return False

    def is_chest_tap(self, hands, pose_landmarks):
        """Chest Tap - Hand tapping chest"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            chest_y = (left_shoulder.y + right_shoulder.y) / 2
            # Allow larger margin for chest tap
            return (abs(left_wrist.y - chest_y) < 0.3 or abs(right_wrist.y - chest_y) < 0.3)
        except:
            return False

    def is_tap_stomach(self, pose_landmarks):
        """Tap Stomach - Hand near stomach"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            stomach_y = (left_hip.y + right_hip.y) / 2
            # Only detect if wrist is clearly below chest and not missing
            if left_wrist.visibility < 0.5 and right_wrist.visibility < 0.5:
                return False
            return (abs(left_wrist.y - stomach_y) < 0.1 or abs(right_wrist.y - stomach_y) < 0.1)
        except:
            return False

    def is_tap_head(self, pose_landmarks):
        """Tap Head - Hand near head"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return (left_wrist.y < nose.y or right_wrist.y < nose.y)
        except:
            return False

    def is_hold_neck(self, pose_landmarks):
        """Hold Neck - Both hands near neck"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            neck = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]  # Approximate
            return (abs(left_wrist.y - neck.y) < 0.1 and abs(right_wrist.y - neck.y) < 0.1)
        except:
            return False

    def is_point_leg(self, pose_landmarks):
        """Point Leg - Wrist near knee"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            return (abs(left_wrist.y - left_knee.y) < 0.1 or abs(right_wrist.y - right_knee.y) < 0.1)
        except:
            return False

    def is_hold_arm(self, pose_landmarks):
        """Hold Arm - Wrists close"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            return ((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)**0.5 < 0.1
        except:
            return False

    def is_hold_shoulder(self, pose_landmarks):
        """Hold Shoulder - Hand on shoulder"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            return (abs(left_wrist.y - left_shoulder.y) < 0.1 or abs(right_wrist.y - right_shoulder.y) < 0.1)
        except:
            return False

    def is_lying_down(self, pose_landmarks):
        """Lying Down - Body horizontal"""
        try:
            left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return abs(left_hip.y - nose.y) < 0.2
        except:
            return False

    def is_cross_arms_above_head(self, hands, pose_landmarks):
        """Cross Arms Above Head"""
        try:
            left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return (abs(left_elbow.x - right_elbow.x) < 0.1 and left_elbow.y < nose.y)
        except:
            return False

    def is_wave_both_hands(self, pose_landmarks):
        """Wave Both Hands - Both hands up"""
        return self.is_hands_up(pose_landmarks)

    def is_kneel_down(self, pose_landmarks):
        """Kneel Down - Knees bent, body lowered"""
        try:
            left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            # Only detect if knees and ankles are visible
            if left_knee.visibility < 0.5 or right_knee.visibility < 0.5 or left_ankle.visibility < 0.5 or right_ankle.visibility < 0.5:
                return False
            return (left_knee.y < left_ankle.y and right_knee.y < right_ankle.y)
        except:
            return False

    def is_crawling_crouch(self, pose_landmarks):
        """Crawling Crouch - Knees bent, body low"""
        try:
            left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return (left_knee.y < nose.y and right_knee.y < nose.y)
        except:
            return False

    def is_running_motion(self, pose_landmarks):
        """Running Motion - Legs apart"""
        try:
            left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            return abs(left_knee.x - right_knee.x) > 0.2
        except:
            return False

    def is_jumping_waving(self, pose_landmarks):
        """Jumping Waving - Hands up and legs close"""
        try:
            hands_up = self.is_hands_up(pose_landmarks)
            left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            legs_close = abs(left_knee.x - right_knee.x) < 0.1
            return hands_up and legs_close
        except:
            return False

    def is_shaking_hands(self, pose_landmarks):
        """Shaking Hands - Hands close"""
        return self.is_hold_arm(pose_landmarks)

    def is_hands_on_hips(self, pose_landmarks):
        """Hands on Hips"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            return (abs(left_wrist.y - left_hip.y) < 0.1 and abs(right_wrist.y - right_hip.y) < 0.1)
        except:
            return False

    def is_touch_forehead_sway(self, pose_landmarks):
        """Touch Forehead Sway - Hand on forehead"""
        try:
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
            return (left_wrist.y < nose.y or right_wrist.y < nose.y)
        except:
            return False

    def is_chest_tap(self, hands, pose_landmarks):
        """Chest Tap - Hand positioned on chest area"""
        try:
            if not pose_landmarks:
                return False
            
            # Get chest position from pose landmarks
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            chest_center_x = (left_shoulder.x + right_shoulder.x) / 2
            chest_center_y = (left_shoulder.y + right_shoulder.y) / 2 + 0.1  # Slightly below shoulders
            
            # Check if any hand is near chest
            for hand_info in hands:
                hand_landmarks = hand_info['landmarks']
                wrist = hand_landmarks.landmark[0]
                
                # Check if hand is in chest area
                distance_to_chest = self.dist(
                    (wrist.x, wrist.y),
                    (chest_center_x, chest_center_y)
                )
                
                if distance_to_chest < 0.15:  # Close to chest
                    return True
            
            return False
        except:
            return False

    def is_arms_crossed(self, hands, pose_landmarks):
        """Arms Crossed - EMERGENCY GESTURE - requires both hands and pose"""
        try:
            if len(hands) < 2 or not pose_landmarks:
                return False
            
            # Get pose landmarks
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            left_elbow = pose_landmarks.landmark[13]
            right_elbow = pose_landmarks.landmark[14]
            
            # Find left and right hands
            left_hand = None
            right_hand = None
            
            for hand_info in hands:
                if hand_info['handedness'] == 'Left':
                    left_hand = hand_info['landmarks']
                elif hand_info['handedness'] == 'Right':
                    right_hand = hand_info['landmarks']
            
            if not left_hand or not right_hand:
                return False
            
            # Check if arms are crossed
            left_wrist = left_hand.landmark[0]
            right_wrist = right_hand.landmark[0]
            
            # Arms crossed: left wrist should be on right side and vice versa
            arms_crossed = left_wrist.x > right_wrist.x
            
            # Check if elbows are at chest level
            chest_level = (0.4 < left_elbow.y < 0.7 and 0.4 < right_elbow.y < 0.7)
            
            # Check if wrists are close to opposite shoulders
            left_to_right_shoulder = abs(left_wrist.x - right_shoulder.x) < 0.2
            right_to_left_shoulder = abs(right_wrist.x - left_shoulder.x) < 0.2
            
            return (arms_crossed and chest_level and 
                   left_to_right_shoulder and right_to_left_shoulder)
        except:
            return False

    def is_cross_arms_above_head(self, hands, pose_landmarks):
        """Cross Arms Above Head - Both arms raised and crossed above head"""
        try:
            if len(hands) < 2 or not pose_landmarks:
                return False
            
            # Get head position (nose landmark)
            nose = pose_landmarks.landmark[0]
            
            # Check if both hands are above head level
            hands_above_head = []
            for hand_info in hands:
                wrist = hand_info['landmarks'].landmark[0]
                if wrist.y < nose.y - 0.1:  # Above head
                    hands_above_head.append(hand_info)
            
            if len(hands_above_head) >= 2:
                # Check if arms are crossed above head
                left_hand = None
                right_hand = None
                
                for hand_info in hands_above_head:
                    if hand_info['handedness'] == 'Left':
                        left_hand = hand_info['landmarks']
                    elif hand_info['handedness'] == 'Right':
                        right_hand = hand_info['landmarks']
                
                if left_hand and right_hand:
                    # Check if hands are crossed
                    left_wrist = left_hand.landmark[0]
                    right_wrist = right_hand.landmark[0]
                    return left_wrist.x > right_wrist.x  # Crossed position
            
            return False
        except:
            return False

    def process_frame(self, frame) -> Dict:
        """Process a single frame for gesture detection"""
        try:
            landmarks_data = {'hands': [], 'pose': None}
            
            # Check if we have MediaPipe available
            if MEDIAPIPE_AVAILABLE and self.holistic:
                # Extract landmarks
                landmarks_data = self.extract_landmarks(frame)
                
                # Detect gesture with smoothing
                gesture, confidence, is_emergency = self.detect_gesture(landmarks_data)
                
                # Apply smoothing for stable detection
                if gesture != 'NONE' and gesture != 'ERROR':
                    self.gesture_buffer.append((gesture, confidence, is_emergency))
                    
                    # Get most stable gesture from buffer
                    if len(self.gesture_buffer) >= 3:
                        gesture_counts = defaultdict(list)
                        for g, c, e in self.gesture_buffer:
                            gesture_counts[g].append((c, e))
                        
                        # Find most frequent gesture
                        most_common = max(gesture_counts.items(), 
                                        key=lambda x: len(x[1]))
                        gesture = most_common[0]
                        avg_conf = sum(c for c, e in most_common[1]) / len(most_common[1])
                        is_emergency = any(e for c, e in most_common[1])
                        confidence = min(avg_conf, 0.95)
            else:
                # Fallback mode - no detection available
                gesture, confidence, is_emergency = 'NONE', 0.0, False
            
            # Log detection
            detection_entry = {
                'timestamp': datetime.now().isoformat(),
                'gesture': gesture,
                'confidence': confidence,
                'is_emergency': is_emergency,
                'hands_detected': len(landmarks_data.get('hands', []))
            }
            
            # Add to logs (only for actual detections)
            if gesture != 'NONE' and gesture != 'ERROR':
                self.detection_log.append(detection_entry)
                if is_emergency:
                    self.emergency_log.append(detection_entry)
                    logger.warning(f"Emergency gesture detected: {gesture} (confidence: {confidence:.3f})")
            
            # Keep logs manageable
            if len(self.detection_log) > 1000:
                self.detection_log = self.detection_log[-500:]
            if len(self.emergency_log) > 100:
                self.emergency_log = self.emergency_log[-50:]
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'is_emergency': is_emergency,
                'hands_detected': len(landmarks_data.get('hands', [])),
                'landmarks': landmarks_data,  # Include landmarks for frontend display
                'timestamp': detection_entry['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'gesture': 'ERROR',
                'confidence': 0.0,
                'is_emergency': False,
                'landmarks': {'hands': [], 'pose': None},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_detection_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent detection logs"""
        return self.detection_log[-limit:] if limit else self.detection_log
    
    def get_emergency_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent emergency gesture logs"""
        return self.emergency_log[-limit:] if limit else self.emergency_log
    
    def clear_logs(self):
        """Clear detection logs"""
        self.detection_log.clear()
        self.emergency_log.clear()
        logger.info("Detection logs cleared")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        total_detections = len(self.detection_log)
        emergency_detections = len(self.emergency_log)
        
        if total_detections == 0:
            return {
                'total_detections': 0,
                'emergency_detections': 0,
                'emergency_rate': 0.0,
                'most_common_gesture': 'None',
                'avg_confidence': 0.0
            }
        
        # Calculate most common gesture
        gesture_counts = {}
        confidence_sum = 0
        
        for log_entry in self.detection_log:
            gesture = log_entry['gesture']
            confidence = log_entry['confidence']
            
            if gesture != 'NONE' and gesture != 'ERROR':
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                confidence_sum += confidence
        
        most_common = max(gesture_counts.items(), key=lambda x: x[1])[0] if gesture_counts else 'None'
        avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0.0
        
        return {
            'total_detections': total_detections,
            'emergency_detections': emergency_detections,
            'emergency_rate': (emergency_detections / total_detections) * 100,
            'most_common_gesture': most_common,
            'avg_confidence': avg_confidence,
            'gesture_distribution': gesture_counts
        }

# Global sign detector instance
sign_detector = None

def initialize_sign_detector():
    """Initialize the global sign detector"""
    global sign_detector
    try:
        sign_detector = SignDetector()
        logger.info("Sign detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize sign detector: {e}")
        return False

def get_sign_detector() -> Optional[SignDetector]:
    """Get the global sign detector instance"""
    return sign_detector

if __name__ == "__main__":
    # Test the sign detector
    detector = SignDetector()
    
    # Test with webcam if available
    try:
        if MEDIAPIPE_AVAILABLE:
            cap = cv2.VideoCapture(0)
            print("Testing sign detection with webcam...")
            print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = detector.process_frame(frame)
                
                # Display result
                if result['gesture'] != 'NONE':
                    status = "🚨 EMERGENCY" if result['is_emergency'] else "✋ GESTURE"
                    print(f"{status}: {result['gesture']} (confidence: {result['confidence']:.3f})")
                
                # Show frame (optional)
                cv2.imshow('Sign Detection Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            stats = detector.get_statistics()
            print(f"\nDetection Statistics:")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Emergency detections: {stats['emergency_detections']}")
            print(f"Most common gesture: {stats['most_common_gesture']}")
        else:
            print("MediaPipe not available - testing initialization only")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Sign detector module created successfully!")