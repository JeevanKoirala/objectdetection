import cv2
import ultralytics
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

def initialize_input():
    input_type = input("Enter input type (camera, image, video): ").strip().lower()
    if input_type == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access camera.")
            exit()
        return cap, None
    elif input_type == "image":
        image_path = input("Enter image path: ").strip()
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image.")
            exit()
        return None, frame
    elif input_type == "video":
        video_path = input("Enter video path: ").strip()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not load video.")
            exit()
        return cap, None
    else:
        print("Invalid input type")
        exit()

def process_frame(frame):
    results = model(frame, conf=0.5, iou=0.4)
    return results[0].boxes.data.cpu().numpy()

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf < 0.5:
            continue
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_video(cap):
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = process_frame(frame)
        draw_detections(frame, detections)
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        max_width = 1024
        max_height = 768
        height, width = frame.shape[:2]

        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Object Detection', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(frame):
    detections = process_frame(frame)
    draw_detections(frame, detections)

    max_width = 1024
    max_height = 768
    height, width = frame.shape[:2]

    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Object Detection', resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap, frame = initialize_input()
if cap:
    process_video(cap)
elif frame is not None:
    process_image(frame)
