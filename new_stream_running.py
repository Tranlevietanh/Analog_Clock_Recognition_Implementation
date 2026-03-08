import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import einops
import torch.nn as nn
import torchvision.models as models
from clock_utils import warp
from collections import deque

YOLO_MODEL_PATH = r"D:\Bai tap\Visual Studio for Python\Midterm\YOLO_customized.pt"
YOLO_POSE_PATH = r"D:\Bai tap\Visual Studio for Python\Midterm\YOLO_pose_customized_2.pt"
CLOCK_CLASS_ID = 0

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

BOX_WINDOW = 3         # Detect 3 frames for box averaging
POSE_WINDOW = 5        # Collect 5 pose angles before rotating/reading
PROCESS_INTERVAL = 1.0 
CROP_MARGIN = 0.12

TIME_MODEL_PATH = r"D:\Bai tap\Visual Studio for Python\Midterm\full_st.pth"
STN_MODEL_PATH  = r"D:\Bai tap\Visual Studio for Python\Midterm\full.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO(YOLO_MODEL_PATH)
yolo_pose = YOLO(YOLO_POSE_PATH)

model_stn = models.resnet50(pretrained=False)
model_stn.fc = nn.Linear(2048, 8)
model_time = models.resnet50(pretrained=False)
model_time.fc = nn.Linear(2048, 720)

model_time.load_state_dict(torch.load(TIME_MODEL_PATH, map_location=device))
model_stn.load_state_dict(torch.load(STN_MODEL_PATH, map_location=device))
model_time.to(device).eval()
model_stn.to(device).eval()

def compute_iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)

def average_box(boxes):
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (int(sum(xs1)/len(xs1)), int(sum(ys1)/len(ys1)), int(sum(xs2)/len(xs2)), int(sum(ys2)/len(ys2)))

def read_clock_time(cropped_img):
    img = cv2.resize(cropped_img, (224, 224)) / 255.0
    img = einops.rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_st = model_stn(img)
        pred_st = torch.cat([pred_st, torch.ones(1, 1, device=device)], dim=1)
        Minv_pred = pred_st.view(-1, 3, 3)
        img_warped = warp(img, Minv_pred)
        pred = model_time(img_warped)
        idx = torch.argmax(pred, dim=1)[0]
        return idx // 60, idx % 60

cap = cv2.VideoCapture(0)
box_buffers = {}      # track_id -> deque of boxes
angle_buffers = {}    # track_id -> deque of angles
clock_results = {}    # track_id -> string time

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    results = yolo.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
    active_ids = set()

    for result in results:
        if result.boxes is None or result.boxes.id is None: continue

        for box, track_id, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().cpu().tolist(), result.boxes.cls.cpu().tolist()):
            if cls != CLOCK_CLASS_ID: continue
            active_ids.add(track_id)

            # 1. BOX AVERAGING (3 Frames)
            if track_id not in box_buffers: box_buffers[track_id] = deque(maxlen=BOX_WINDOW)
            box_buffers[track_id].append(tuple(map(int, box)))
            
            if len(box_buffers[track_id]) < BOX_WINDOW: continue
            
            ax1, ay1, ax2, ay2 = average_box(box_buffers[track_id])
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)

            # 2. POSE ANGLE CALCULATION
            bw, bh = ax2 - ax1, ay2 - ay1
            x1m, y1m = max(0, int(ax1 - CROP_MARGIN * bw)), max(0, int(ay1 - CROP_MARGIN * bh))
            x2m, y2m = min(w, int(ax2 + CROP_MARGIN * bw)), min(h, int(ay2 + CROP_MARGIN * bh))
            crop = frame[y1m:y2m, x1m:x2m]

            pose_res = yolo_pose(crop, conf=0.5, verbose=False)
            if pose_res and pose_res[0].keypoints is not None and len(pose_res[0].keypoints.xy[0]) >= 5:
                pts = pose_res[0].keypoints.xy[0].cpu().numpy()
                c_x, c_y = pts[0]
                marks = [(1, 90), (2, 270), (3, 0), (4, 180)]
                current_angles = []
                for i, target in marks:
                    if pts[i][0] > 0:
                        dx, dy = pts[i][0] - c_x, c_y - pts[i][1]
                        measured = np.degrees(np.arctan2(dy, dx))
                        current_angles.append(((target - measured + 180) % 360) - 180)
                
                if current_angles:
                    if track_id not in angle_buffers: angle_buffers[track_id] = deque(maxlen=POSE_WINDOW)
                    angle_buffers[track_id].append(np.mean(current_angles))

            # 3. ROTATE & READ (Only after 5 angles collected)
            if track_id in angle_buffers and len(angle_buffers[track_id]) == POSE_WINDOW:
                avg_angle = np.mean(angle_buffers[track_id])
                
                ch, cw = crop.shape[:2]
                M = cv2.getRotationMatrix2D((cw//2, ch//2), avg_angle, 1.0)
                aligned = cv2.warpAffine(crop, M, (cw, ch))
                
                hour, minute = read_clock_time(aligned)
                clock_results[track_id] = f"{hour:02d}:{minute:02d}"

            if track_id in clock_results:
                cv2.putText(frame, clock_results[track_id], (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    for tid in list(box_buffers.keys()):
        if tid not in active_ids:
            box_buffers.pop(tid, None)
            angle_buffers.pop(tid, None)
            clock_results.pop(tid, None)

    cv2.imshow("Auto Clock Reader System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()

cv2.destroyAllWindows()
