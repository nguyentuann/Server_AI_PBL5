import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constant.important_keypoint import IMPORTANT_KP


def convertData(keypoints_data):
    try:
        features = []
        for kp in IMPORTANT_KP:
            point = keypoints_data.get(kp, None)
            if point is not None:
                x, y, z, visibility = (
                    point["x"],
                    point["y"],
                    point["z"],
                    point["visibility"],
                )
                features.extend([x, y, z, visibility])
        return np.array(features).reshape(1, -1)    
    except Exception as e:
        print(f"Error in convertData: {e}")
        return None
