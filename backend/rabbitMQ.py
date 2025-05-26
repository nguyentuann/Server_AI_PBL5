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

# ip_server_backend = "amqps://uwsuamrb:nXsf-6FMy-ePOZhKq4TyfWOH4h0YB1Rq@fuji.lmq.cloudamqp.com/uwsuamrb"
ip_server_backend = "192.168.35.214"

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

user_reps = {} # ! mảng số lần squat của user
user_labels = {} # ! mảng các lỗi trong rep -> lấy ra ảnh lỗi
user_session_id = {} # ! mảng các session_id của user -> mảng các session id 

def get_most_common_label(labels_count):
    """Tìm nhãn phổ biến nhất theo logic đặc biệt của bài toán."""
    if not labels_count:
        return CORRECT
    counter = Counter(labels_count)
    total = sum(counter.values())
    # Ưu tiên lỗi ERROR_BACK_BEND nếu >= 3 lần
    if counter.get(ERROR_BACK_BEND, 0) >= 3:
        return ERROR_BACK_BEND
    most_common = counter.most_common()
    first_label, first_count = most_common[0]
    if first_label == CORRECT and len(most_common) > 1:
        second_label, second_count = most_common[1]
        if (second_count / total) > 0.4:
            return second_label
        return first_label
    return first_label


def process_keypoints(ch, method, props, body):
    try:
        message = body.decode()
        data = json.loads(message)
        keypoints = data.get("key_points")
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        if not keypoints or not user_id or not session_id:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # Nếu user có session cũ và session khác session hiện tại => reset dữ liệu
        prev_session = user_session_id.get(user_id)
        if prev_session is not None and prev_session != session_id:
            print(f"🔄 Session mới phát hiện cho user {user_id}, reset dữ liệu.")
            user_reps[user_id] = 0
            user_labels[user_id] = []

        # Cập nhật session hiện tại 
        user_session_id[user_id] = session_id

        # Khởi tạo mảng nếu chưa có
        labels_count = user_labels.setdefault(user_id, [])
        count = user_reps.get(user_id, 0)

        features = convertData(keypoints)

        label_error = squat_model.predict(features)
        labels_count.append(label_error)

        current_state, is_counted = squat_count.predict(features)

        if is_counted:
            count += 1
            user_reps[user_id] = count

            most_common_label = get_most_common_label(labels_count)

            if most_common_label != CORRECT:
                error_message = labels_dict.get(most_common_label, "Unknown")
                print(f"⚠️ Lỗi phổ biến: {error_message}")
                print(f"🧠 labels_count: {labels_count}")

                middle_idx = None
                if most_common_label in labels_count:
                    label_indices = [i for i, label in enumerate(labels_count) if label == most_common_label]
                    if label_indices:
                        mid_point = len(labels_count) // 2 + 2
                        middle_idx = min(label_indices, key=lambda idx: abs(idx - mid_point))
                        print(f"📊 Vị trí giữa nhất của lỗi {error_message}: {middle_idx}")

                result = {
                    "rep_index": count,
                    "content": error_message,
                    "time": time.ctime(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "image_id": middle_idx,
                }

                # Gửi message tới RabbitMQ kết quả lỗi
                ch.basic_publish(
                    exchange=RESULT_EXCHANGE_NAME,
                    routing_key=RESULT_ROUTING,
                    body=json.dumps(result),
                )
                user_labels[user_id] = []  # reset mảng lỗi sau khi gửi
            else:
                print("✅ Không phát hiện lỗi, rep hợp lệ.")
                # Không reset mảng lỗi ở đây vì có thể lỗi xuất hiện trong các rep tiếp theo

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Lỗi xử lý message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


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

     
        
        # bắt đầu nhận dữ liệu từ processing.queue
        channel.basic_consume(
            queue=PROCESSING_QUEUE, on_message_callback=process_keypoints
        )

        channel.start_consuming()
    except KeyboardInterrupt:
        print("🛑 Ngắt kết nối RabbitMQ")
    except Exception as e:
        print(f"❌ Lỗi kết nối RabbitMQ: {e}")
