import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constant.important_keypoint import IMPORTANT_KP


def convertData(keypoints_data):
    try:
        features = []

        # Lấy tọa độ x của NOSE để làm chuẩn dịch chuyển
        nose = keypoints_data.get("NOSE", None)
        if nose is None:
            raise ValueError("NOSE keypoint is missing")

        delta_x = nose["x"] - 0.5

        for kp in IMPORTANT_KP:
            point = keypoints_data.get(kp, None)
            if point is not None:
                x = point["x"] - delta_x  # dịch chuyển theo mốc NOSE
                y = point["y"]
                z = point["z"]
                visibility = point["visibility"]
                features.extend([x, y, z, visibility])
            else:
                # Nếu thiếu điểm, thêm 4 giá trị 0
                features.extend([0, 0, 0, 0])

        return np.array(features).reshape(1, -1)

    except Exception as e:
        print(f"Error in convertData: {e}")
        return None
