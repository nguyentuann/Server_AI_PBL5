import os
from constant.important_keypoint import IMPORTANT_KP

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import pandas as pd


class SquatCountModel:
    def __init__(
        self,
        scaler_path="UP_DOWN/scaler.pkl",
        model_path="UP_DOWN/LR_Up_Down_model.pkl",
        important_kp=None,
    ):
        self.pre_pre_state = "Up"
        self.pre_state = "Up"

        self.scaler = self.load_scaler(scaler_path)
        self.model = self.load_model(model_path)

        if important_kp is None:
            important_kp = IMPORTANT_KP
        self.headers = self._generate_headers(important_kp)

    def load_scaler(self, path):
        """Load scaler từ file pickle"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_model(self, path):
        """Load mô hình ML (pickle)"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _generate_headers(self, keypoints):
        """Tạo header chuẩn theo keypoint đầu vào"""
        headers = []
        for kp in keypoints:
            headers.extend(
                [
                    f"{kp.lower()}_x",
                    f"{kp.lower()}_y",
                    f"{kp.lower()}_z",
                    f"{kp.lower()}_v",
                ]
            )
        return headers

    def predict(self, features):
        """Dự đoán trạng thái squat (Up/Down)"""
        df = pd.DataFrame(features, columns=self.headers)
        features_scaled = self.scaler.transform(df)
        prediction = self.model.predict(features_scaled)

        current_state = "Up" if prediction[0] == 1 else "Down"

        # Xác định nếu đây là một lần squat hợp lệ
        is_counted = (
            self.pre_pre_state == "Up"
            and self.pre_state == "Down"
            and current_state == "Up"
        )

        # Cập nhật trạng thái
        if current_state != self.pre_state:
            self.pre_pre_state = self.pre_state
            self.pre_state = current_state

        return current_state, is_counted


# Khởi tạo model để dùng trong app
squat_count = SquatCountModel()
