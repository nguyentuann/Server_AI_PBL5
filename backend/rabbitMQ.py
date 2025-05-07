import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json
import pika
import time
import numpy as np
from collections import Counter

from AI.model_detection import squat_model
from AI.model_count import squat_count
from AI.convert_data import convertData
from constant.labels import labels_dict

ip_serverAI = "192.168.1.36"

CORRECT = np.int64(0)
ERROR_BACK_BEND = np.int64(5)

# RabbitMQ cấu hình

# backend gửi
PROCESSING_EXCHANGE_NAME = "gympose.ai.process.direct"
PROCESSING_QUEUE = "gympose.ai.process.ai-server"

# serverAI gửi
RESULT_ROUTING = "ai.result.backend"
RESULT_QUEUE = "gympose.ai.result.backend"
RESULT_EXCHANGE_NAME = "gympose.ai.result.direct"

user_reps = {}


def process_keypoints(ch, method, props, body):
    labels_count = []

    try:
        message = body.decode()
        json_message = json.loads(message)
        keypoints = json_message["key_points"]
        user_id = json_message["user_id"]

        count = user_reps.get(user_id, 0)

        features = convertData(keypoints)

        # Dự đoán lỗi
        label_error = squat_model.predict(features)
        print(f"🧠 Lỗi dự đoán: {label_error}")
        labels_count.append(label_error)

        # Dự đoán trạng thái + đếm rep
        current_state, is_counted = squat_count.predict(features)
        if is_counted:
            count += 1
            user_reps[user_id] = count

            most_common_label = CORRECT
            if labels_count:
                counter = Counter(labels_count)
                total = sum(counter.values())

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

            print(f"🧠 Lỗi phổ biến: {most_common_label}")

            error_message = labels_dict.get(most_common_label, "Unknown")
            print(f"⚠️ Lỗi phổ biến: {error_message}")

            result = {
                "rep_num": count,
                "content": error_message,
                "time": time.ctime(),
                "user_id": user_id,
            }

            # Gửi kết quả về result.queue thông qua exchange
            ch.basic_publish(
                exchange=RESULT_EXCHANGE_NAME,
                routing_key=RESULT_ROUTING,
                body=json.dumps(result),
            )
            labels_count.clear()
            return

        # Xác nhận đã xử lý
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Lỗi xử lý message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_server():
    print("🚀 Server AI sẵn sàng chờ dữ liệu...")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=ip_serverAI))
    channel = connection.channel()

    # bắt đầu nhận dữ liệu từ processing.queue
    channel.basic_consume(queue=PROCESSING_QUEUE, on_message_callback=process_keypoints)

    channel.start_consuming()
