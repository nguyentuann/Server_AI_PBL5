import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from collections import Counter
from fastapi import WebSocket
from AI.model_detection import squat_model
from AI.model_count import squat_count
from AI.convert_data import convertData
from constant.labels import labels_dict

CORRECT = np.int64(0)
ERROR_BACK_BEND = np.int64(5)


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    label_counts = []
    count = 0

    while True:
        try:
            keypoints_data = await websocket.receive_text()
            keypoints_data = json.loads(keypoints_data)

            if keypoints_data.get("keypoints") is not None:
                features_raw = keypoints_data["keypoints"]

                features = convertData(features_raw)

                # Dự đoán lỗi
                label = squat_model.predict(features)
                print(f"loi:  {label}")
                label_counts.append(label)

                # Dự đoán đếm squat
                current_state, is_counted = squat_count.predict(features)
                if is_counted:
                    count += 1

                    # Xử lý lỗi phổ biến nhất trong chuỗi
                    if label_counts:

                        counter = Counter(label_counts)
                        total = sum(counter.values())
                        print(f"total: {total}")

                        if counter.get(ERROR_BACK_BEND, 0) >= 3:
                            most_common_label = ERROR_BACK_BEND

                        else:
                            most_common = counter.most_common()

                            first_label, _ = most_common[0]

                            if first_label == CORRECT and len(most_common) > 1:
                                second_label, second_count = most_common[1]

                                if (second_count / total) > 0.4:
                                    most_common_label = second_label
                                else:
                                    most_common_label = first_label
                            else:
                                most_common_label = first_label

                        error_message = labels_dict.get(most_common_label, "Unknown")
                        print(f"⚠️ Lỗi: {error_message}")

                        # Gửi phản hồi về client
                        await websocket.send_text(
                            json.dumps({"repNum": count, "content": error_message})
                        )

                        label_counts.clear()

        except Exception as e:
            print("⚠️ Lỗi kết nối:", e)
            break
