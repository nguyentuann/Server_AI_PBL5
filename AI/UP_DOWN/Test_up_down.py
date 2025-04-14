import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.layers import Layer  # type: ignore
import tensorflow.keras.backend as K  # type: ignore

# Tải cả mô hình và scaler
with open("LR_Up_Down_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Danh sách keypoints quan trọng
IMPORTANT_KP = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

# Create column names that match those used during training
headers = []
for kp in IMPORTANT_KP:
    headers.extend(
        [f"{kp.lower()}_x", f"{kp.lower()}_y", f"{kp.lower()}_z", f"{kp.lower()}_v"]
    )

# State
UP = 1
DOWN = 0


# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Đọc video đầu vào
cap = cv2.VideoCapture(1)

# Lấy thông tin video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo VideoWriter để lưu video đầu ra
output_path = "Demo/Dem_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Các biến trạng thái
pre_state = "Up"
pre_pre_state = "Up"
pre_time = datetime(1970, 1, 1, 0, 0, 0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Total count: {count}")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Kiểm tra visibility > 0.8 cho tất cả 9 điểm
        if all(
            landmarks[getattr(mp_pose.PoseLandmark, kp)].visibility > 0.8
            for kp in IMPORTANT_KP
        ):
            # Chỉ xử lý nếu đủ điều kiện visibility

            features = []
            for kp in IMPORTANT_KP:
                landmark = getattr(mp_pose.PoseLandmark, kp)
                features.extend(
                    [
                        landmarks[landmark].x,
                        landmarks[landmark].y,
                        landmarks[landmark].z,
                        landmarks[landmark].visibility,
                    ]
                )

            features_array = np.array(features).reshape(1, -1)
            features_df = pd.DataFrame(features_array, columns=headers)
            features_scaled = scaler.transform(features_df)

            label_array = model.predict(features_scaled)
            label = label_array[0]

            labels_dict = {0: "Down", 1: "Up"}
            label_text = labels_dict.get(label, "Unknown")

            now_time = datetime.now()

            # Chỉ xử lý nếu đã qua 0.1 giây (lọc nhiễu)
            if (now_time - pre_time).total_seconds() > 0.3:
                if label_text != pre_state:
                    print(f"Frame: {label_text}")  # chỉ in khi trạng thái thay đổi

                    # Đếm khi chuyển từ Down -> Up
                    # Chỉ đếm nếu trạng thái: Up (hiện tại), Down (trước đó), Up (trước trước nữa)
                    if (
                        pre_pre_state == "Up"
                        and pre_state == "Down"
                        and label_text == "Up"
                    ):
                        count += 1
                        print(f"Count: {count}")

                    # Cập nhật trạng thái
                    pre_pre_state = pre_state
                    pre_state = label_text
                    pre_time = now_time

            # Hiển thị nhãn lên frame
            cv2.putText(
                frame,
                f"Prediction: {label_text}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                f"Count: {count}",
                (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

    # Ghi và hiển thị video
    out.write(frame)
    cv2.imshow("Squat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
