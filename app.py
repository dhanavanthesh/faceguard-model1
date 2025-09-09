import base64
import io
import math
from datetime import datetime
from collections import deque, defaultdict

import cv2
import numpy as np
from PIL import Image
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# ---------- Config ----------
HOST = "0.0.0.0"
PORT = 5000
SMOOTHING_WINDOW = 5
MOTION_WINDOW = 12
# ----------------------------

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# MediaPipe setups
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --- Drawing connections ---
POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# --- Client buffers ---
client_buffers = {}
def ensure_client_buffers(sid):
    if sid not in client_buffers:
        client_buffers[sid] = {
            "left": deque(maxlen=SMOOTHING_WINDOW),
            "right": deque(maxlen=SMOOTHING_WINDOW),
            "events": deque(maxlen=SMOOTHING_WINDOW),
            "left_wrist_x": deque(maxlen=MOTION_WINDOW),
            "right_wrist_x": deque(maxlen=MOTION_WINDOW),
            "left_wrist_y": deque(maxlen=MOTION_WINDOW),
            "right_wrist_y": deque(maxlen=MOTION_WINDOW),
            "help_flags": deque(maxlen=6),
            "help_confs": deque(maxlen=6),
            "fire_flags": deque(maxlen=6),
            "fire_confs": deque(maxlen=6),
            "sos_flags": deque(maxlen=6),
            "sos_confs": deque(maxlen=6),
        }
    return client_buffers[sid]

# --- Utils ---
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle(a, b, c):
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.hypot(ab[0], ab[1]) * math.hypot(cb[0], cb[1]) + 1e-6
    cosv = max(-1.0, min(1.0, dot/mag))
    return math.degrees(math.acos(cosv))

def smooth_label_conf(sequence):
    if not sequence:
        return ("none", 0.0)
    tally = defaultdict(list)
    for lab, conf in sequence:
        tally[lab].append(conf)
    best_lab = max(tally.keys(), key=lambda k: (len(tally[k]), sum(tally[k])/max(1,len(tally[k]))))
    avg_conf = sum(tally[best_lab]) / max(1, len(tally[best_lab]))
    return (best_lab, float(avg_conf))

def smooth_events(sequence_of_lists):
    tally = defaultdict(list)
    total_frames = len(sequence_of_lists)
    for lst in sequence_of_lists:
        for name, conf in lst:
            tally[name].append(conf)
    min_count = max(2, int(0.4 * max(1, total_frames)))
    min_conf = 0.8
    ranked = sorted(((name, sum(vals)/len(vals), len(vals)) for name, vals in tally.items()), key=lambda x: (x[2], x[1]), reverse=True)
    result = []
    for name, avg_conf, cnt in ranked:
        if cnt >= min_count and avg_conf >= min_conf:
            result.append({"name": name, "confidence": float(avg_conf)})
        if len(result) >= 3:
            break
    return result

# --- Hand gesture classification ---
def classify_hand_gesture(hand_landmarks, handedness_label):
    if hand_landmarks is None:
        return ("none", 0.0)
    lm = hand_landmarks.landmark
    finger_tips = {"index":8, "middle":12, "ring":16, "pinky":20}
    finger_pips = {"index":6, "middle":10, "ring":14, "pinky":18}
    ext = {f: lm[finger_tips[f]].y < lm[finger_pips[f]].y for f in finger_tips}
    thumb_tip = lm[4]
    wrist = lm[0]
    angle_thumb = angle((lm[3].x, lm[3].y), (lm[2].x, lm[2].y), (lm[5].x, lm[5].y))
    extended_count = sum(int(v) for v in ext.values())
    avg_tip_wrist = np.mean([dist((lm[i].x, lm[i].y),(wrist.x,wrist.y)) for i in [8,12,16,20]])
    fist_threshold = 0.06
    open_threshold = 0.18

    if ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return ("pointing_up", 0.88)
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return ("victory", 0.92)
    if ((lm[4].x - lm[3].x)**2 + (lm[4].y - lm[3].y)**2) > 0.0004 and ext["pinky"] and (not ext["middle"] and not ext["ring"]):
        return ("i_love_you", 0.90)
    if (not ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]) and angle_thumb > 25:
        return ("thumbs_up", 0.94) if thumb_tip.y < wrist.y else ("thumbs_down", 0.90)
    if extended_count >= 4 and avg_tip_wrist > open_threshold:
        return ("open_palm", 0.93)
    if avg_tip_wrist < fist_threshold:
        return ("closed_fist", 0.9)
    return ("unknown", 0.25)

# --- High-level detection ---
def detect_high_level(landmarks_pose, left_hand_gesture, right_hand_gesture):
    detections = []
    conf = 0.7
    if landmarks_pose:
        try:
            l_sh, r_sh = landmarks_pose.landmark[11], landmarks_pose.landmark[12]
            l_wr, r_wr = landmarks_pose.landmark[15], landmarks_pose.landmark[16]
            if getattr(l_sh,"visibility",1.0)>0.5 and getattr(l_wr,"visibility",1.0)>0.5:
                if l_wr.y < l_sh.y - 0.03: detections.append(("left_hand_up",conf))
            if getattr(r_sh,"visibility",1.0)>0.5 and getattr(r_wr,"visibility",1.0)>0.5:
                if r_wr.y < r_sh.y - 0.03: detections.append(("right_hand_up",conf))
        except: pass
    if left_hand_gesture[0]=="open_palm": detections.append(("stop", max(conf,left_hand_gesture[1])))
    if right_hand_gesture[0]=="open_palm": detections.append(("stop", max(conf,right_hand_gesture[1])))
    return detections

# --- Motion / Emergency events ---
# --- Motion / Emergency events ---
def detect_motion_events(landmarks_pose, left_g, right_g):
    events=[]
    if landmarks_pose:
        try:
            l_sh, r_sh = landmarks_pose.landmark[11], landmarks_pose.landmark[12]
            l_el, r_el = landmarks_pose.landmark[13], landmarks_pose.landmark[14]
            l_wr, r_wr = landmarks_pose.landmark[15], landmarks_pose.landmark[16]
            shoulder_width = max(1e-3, abs(r_sh.x-l_sh.x))

            # Fire alarm: both hands above head and open palms
            if left_g[0]=="open_palm" and right_g[0]=="open_palm":
                if l_wr.y<l_sh.y-0.02 and r_wr.y<r_sh.y-0.02 and l_el.y<l_sh.y-0.02 and r_el.y<r_sh.y-0.02:
                    events.append(("fire_alarm",0.94))

            # Help: both thumbs up
            if left_g[0]=="thumbs_up" and right_g[0]=="thumbs_up":
                events.append(("help", max(0.9,min(left_g[1],right_g[1]))))

            # SOS: crossed arms (refined)
            dx_l, dy_l = abs(l_wr.x - r_sh.x), abs(l_wr.y - r_sh.y)
            dx_r, dy_r = abs(r_wr.x - l_sh.x), abs(r_wr.y - l_sh.y)
            chest_y = (l_sh.y + r_sh.y) / 2
            elbows_close = abs(l_el.x - r_el.x) < 0.25*shoulder_width
            wrists_crossed = dx_l < 0.3*shoulder_width and dx_r < 0.3*shoulder_width
            if elbows_close and wrists_crossed and abs(l_el.y - chest_y) < 0.2 and abs(r_el.y - chest_y) < 0.2:
                events.append(("sos", 0.95))

            # Chest Pain: hands together on chest
            mid_chest_x = (l_sh.x + r_sh.x) / 2
            mid_chest_y = (l_sh.y + r_sh.y) / 2
            wrists_close = dist((l_wr.x,l_wr.y),(r_wr.x,r_wr.y)) < 0.25*shoulder_width
            wrists_on_chest = abs((l_wr.x+r_wr.x)/2 - mid_chest_x) < 0.2*shoulder_width and abs((l_wr.y+r_wr.y)/2 - mid_chest_y) < 0.2
            if wrists_close and wrists_on_chest:
                events.append(("chest_pain", 0.93))

        except: pass
    return events

def persistent_emergency_events(buf, frame_events):
    names = [e[0] for e in frame_events]
    confs = {e[0]: e[1] for e in frame_events}
    buf["help_flags"].append("help" in names)
    buf["help_confs"].append(confs.get("help",0.0))
    buf["fire_flags"].append("fire_alarm" in names)
    buf["fire_confs"].append(confs.get("fire_alarm",0.0))
    buf["sos_flags"].append("sos" in names)
    buf["sos_confs"].append(confs.get("sos",0.0))
    events=[]
    def stable(flag_deque, conf_deque, need=4):
        trues=sum(1 for v in flag_deque if v)
        avg_conf=sum(conf_deque)/max(1,len(conf_deque))
        return (trues>=need, avg_conf)
    sos_ok,sos_conf=stable(buf["sos_flags"],buf["sos_confs"])
    fire_ok,fire_conf=stable(buf["fire_flags"],buf["fire_confs"])
    help_ok,help_conf=stable(buf["help_flags"],buf["help_confs"])
    if sos_ok: events.append(("sos",max(0.9,sos_conf)))
    elif fire_ok: events.append(("fire_alarm",max(0.9,fire_conf)))
    elif help_ok: events.append(("help",max(0.9,help_conf)))
    return events

# --- Format landmarks for frontend ---
def format_landmarks(landmarks, w, h):
    if not landmarks: return []
    return [(lm.x*w, lm.y*h) for lm in landmarks.landmark]

def format_connections(landmarks, connections, w, h):
    if not landmarks: return []
    pts = [(lm.x*w, lm.y*h) for lm in landmarks.landmark]
    lines = []
    for (i,j) in connections:
        if i < len(pts) and j < len(pts):
            lines.append((pts[i][0], pts[i][1], pts[j][0], pts[j][1]))
    return lines

# --- SocketIO ---
@socketio.on("connect")
def on_connect():
    print("Client connected")
    emit("server_ready",{"msg":"ready"})

@socketio.on("frame")
def on_frame(data):
    try:
        img_b64 = data.get("image",None)
        if img_b64 is None: return
        header, encoded = img_b64.split(",",1) if "," in img_b64 else (None,img_b64)
        frame_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        frame = np.array(img)[:, :, ::-1]
        frame = cv2.resize(frame, (480,360))
        h, w, _ = frame.shape

        results = holistic.process(frame)

        left_g = classify_hand_gesture(results.left_hand_landmarks, "Left") if results.left_hand_landmarks else ("none",0.0)
        right_g = classify_hand_gesture(results.right_hand_landmarks, "Right") if results.right_hand_landmarks else ("none",0.0)

        detections = detect_high_level(results.pose_landmarks, left_g, right_g)
        frame_emergencies = detect_motion_events(results.pose_landmarks, left_g, right_g)
        buf = ensure_client_buffers(request.sid)
        gated_emergencies = persistent_emergency_events(buf, frame_emergencies)
        non_emerg = [(n,c) for (n,c) in detections if n not in ("help","fire_alarm","sos")]
        detections = non_emerg + gated_emergencies

        buf["left"].append(left_g)
        buf["right"].append(right_g)
        buf["events"].append(detections)
        left_g_s = smooth_label_conf(list(buf["left"]))
        right_g_s = smooth_label_conf(list(buf["right"]))
        detections_s = smooth_events(list(buf["events"]))

        resp = {
            "hand_gestures":{
                "left":{"gesture":left_g_s[0],"confidence":float(left_g_s[1])},
                "right":{"gesture":right_g_s[0],"confidence":float(right_g_s[1])}
            },
            "detections": detections_s if detections_s else [{"name":d[0],"confidence":float(d[1])} for d in detections],
            "frame_size":{"width":w,"height":h},
            "pose_points": format_landmarks(results.pose_landmarks, w, h),
            "skeleton_lines": format_connections(results.pose_landmarks, POSE_CONNECTIONS, w, h),
            "left_hand_points": format_landmarks(results.left_hand_landmarks, w, h),
            "left_hand_lines": format_connections(results.left_hand_landmarks, HAND_CONNECTIONS, w, h),
            "right_hand_points": format_landmarks(results.right_hand_landmarks, w, h),
            "right_hand_lines": format_connections(results.right_hand_landmarks, HAND_CONNECTIONS, w, h),
            "timestamp": datetime.utcnow().isoformat()+"Z"
        }
        emit("detection", resp)
    except Exception as e:
        print("Error processing frame:", e)
        emit("error",{"msg":str(e)})

@app.route("/")
def index():
    return send_from_directory("../frontend","index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("../frontend",filename)

if __name__=="__main__":
    print(f"Starting backend on http://{HOST}:{PORT}")
    socketio.run(app, host=HOST, port=PORT)
