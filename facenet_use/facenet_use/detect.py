import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化 MTCNN 和 InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 载入之前保存的特征向量
saved_embeddings = np.load('face_embeddings.npy')

camera = cv2.VideoCapture(0)

try:
    while True:
        # 从摄像头读取一帧
        ret, frame = camera.read()
        if not ret:
            break

        # 检测人脸
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            faces = mtcnn(frame)
            if faces is not None:
                new_embeddings = resnet(faces).detach().cpu().numpy()

                # 对每一个检测到的人脸进行比较
                for i, new_embedding in enumerate(new_embeddings):
                    # 计算与已保存特征向量的余弦距离
                    distances = [cosine(new_embedding, emb) for emb in saved_embeddings]
                    min_distance = min(distances)
                    min_index = distances.index(min_distance)

                    # 设定阈值和显示匹配结果
                    if min_distance < 0.2:
                        match_message = f"Match found: Distance {min_distance:.2f} at index {min_index}"
                    else:
                        match_message = "No match found"

                    # 在视频中标出人脸和匹配结果
                    box = boxes[i]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, match_message, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Face Detection and Recognition', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
