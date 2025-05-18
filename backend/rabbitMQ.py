import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json
import pika
import time
import threading
import numpy as np
from collections import Counter

from AI.model_detection import squat_model
from AI.model_count import squat_count
from AI.convert_data import convertData
from constant.labels import labels_dict

ip_server_backend = "amqps://uwsuamrb:nXsf-6FMy-ePOZhKq4TyfWOH4h0YB1Rq@fuji.lmq.cloudamqp.com/uwsuamrb"
# ip_server_backend = "192.168.148.149"

CORRECT = np.int64(0)
ERROR_BACK_BEND = np.int64(5)
TIMEOUT_SECONDS = 30

# RabbitMQ cấu hình

# backend gửi
PROCESSING_EXCHANGE_NAME = "gympose.ai.process.direct"
PROCESSING_QUEUE = "gympose.ai.process.ai-server"

# serverAI gửi
RESULT_ROUTING = "ai.result.backend"
RESULT_QUEUE = "gympose.ai.result.backend"
RESULT_EXCHANGE_NAME = "gympose.ai.result.direct"

user_reps = {}
user_labels = {}
user_last_data = {}
 

def process_keypoints(ch, method, props, body):

    try:
        message = body.decode()
        json_message = json.loads(message)
        keypoints = json_message["key_points"]
        user_id = json_message["user_id"]
        
        labels_count = user_labels.setdefault(user_id, [])
        
        user_last_data[user_id] = time.time()
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

            if most_common_label != CORRECT:

                error_message = labels_dict.get(most_common_label, "Unknown")
                print(f"⚠️ Lỗi phổ biến: {error_message}")
                
                # lấy labels_count giữa nhất giống với most_common_label
                print(f"🧠 labels_count: {labels_count}")
                middle_idx = None
                if most_common_label in labels_count:
                    # Tìm tất cả vị trí của most_common_label trong labels_count
                    label_indices = [i for i, label in enumerate(labels_count) if label == most_common_label]
                    if label_indices:
                        # Tìm vị trí ở giữa nhất
                        mid_point = len(labels_count) // 2
                        middle_idx = min(label_indices, key=lambda idx: abs(idx - mid_point))
                        print(f"📊 Vị trí giữa nhất của lỗi {error_message}: {middle_idx}")

                result = {
                    "rep_num": count,
                    "content": error_message,
                    "time": time.ctime(),
                    "user_id": user_id,
                    # "image_id": middle_idx,
                }
                

                # Gửi kết quả về result.queue thông qua exchange
                ch.basic_publish(
                    exchange=RESULT_EXCHANGE_NAME,
                    routing_key=RESULT_ROUTING,
                    body=json.dumps(result),
                )
                user_labels.pop(user_id, None)
                return
            else:
                error_message = labels_dict.get(most_common_label, "Unknown")
                print(f"⚠️ Lỗi phổ biến: {error_message}")

        # Xác nhận đã xử lý
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Lỗi xử lý message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def cleanup_inactive_users():
    while True:
        current_time = time.time()
        inactive_users = []

        for user_id, last_seen in list(user_last_data.items()):
            if current_time - last_seen > TIMEOUT_SECONDS:
                inactive_users.append(user_id)

        for user_id in inactive_users:
            print(f"⏱️ User {user_id} không hoạt động > 30st. Reset count.")
            user_reps.pop(user_id, None)
            user_last_data.pop(user_id, None)

        time.sleep(10)  # kiểm tra mỗi 10 giây
        

def start_server():
    connection = None
    channel = None

    try:
        print("🚀 Server AI sẵn sàng chờ dữ liệu...")
        # parameters = pika.URLParameters(ip_server_backend)
        parameters = pika.ConnectionParameters(
            host=ip_server_backend,
            port=5672,  # Port mặc định của RabbitMQ
            virtual_host="/",  # Virtual host mặc định
            credentials=pika.PlainCredentials("guest", "guest")  # Hoặc thông tin tài khoản thật nếu có
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        print("chay luong reset user")
        threading.Thread(target=cleanup_inactive_users, daemon=True).start()
        
        # bắt đầu nhận dữ liệu từ processing.queue
        channel.basic_consume(
            queue=PROCESSING_QUEUE, on_message_callback=process_keypoints
        )

        channel.start_consuming()
    except KeyboardInterrupt:
        print("🛑 Ngắt kết nối RabbitMQ")
    except Exception as e:
        print(f"❌ Lỗi kết nối RabbitMQ: {e}")
