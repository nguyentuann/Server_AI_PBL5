import pika
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

ip_serverAI = "192.168.1.2"

CORRECT = np.int64(0)
ERROR_BACK_BEND = np.int64(5)


# nhận dữ liệu từ backend
def process_keypoints(
    ch,  # channel: đối tượng kết nối hiện tại đến RabbitMQ
    method,  # thông tin metadata của mesage
    props,  # thuộc tính của message (reply_to, correlation_id, ...)
    body,  # nội dung của message
):

    labels_count = []
    count = 0

    features_raw = body.decode()  #

    print(f"nhan tu backend: {features_raw}")

    try:
        features = convertData(features_raw)

        # dự đoán lỗi
        label_error = squat_model.predict(features)
        print(f"loi:  {label_error}")

        # dự đoán đếm squat
        current_state, is_counted = squat_count.predict(features)
        if is_counted:
            count += 1

            # chọn lỗi phổ biến nhất
            if labels_count:

                counter = Counter(labels_count)
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
                        pass

                error_message = labels_dict.get(most_common_label, "Unknown")
                print(f"⚠️ Lỗi: {error_message}")

                response = json.dumps({"repNum": count, "content": error_message})
                send_result(ch, props, response)
                labels_count.clear()

        # xác nhận đã xử lý message thành công, xóa khỏi queue
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(e)


def send_result(ch, props, body):
    ch.basic_publish(
        exchange="",  # exchange mặc định là direct
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id,  # để bên A biết msg phản hồi nào khới với request nào
        ),
        body=body,
    )


def start_server():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=ip_serverAI,
        ),
    )
    channel = connection.channel()

    channel.queue_declare(queue="result_queue")
    channel.basic_consume(queue="result_queue", on_message_callback=process_keypoints)

    channel.start_consuming()


if "__name__" == "__main__":
    start_server()
