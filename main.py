import cv2
import numpy as np
import dlib


# 关键点128D编码
def get_128d_features(image, detector, predictor, encoder, upsample=1, jet=1):
    faces = detector(image, upsample)
    faces_keypoints = [predictor(image, face) for face in faces]
    return [np.array(encoder.compute_face_descriptor(image, face_keypoints, jet)) for face_keypoints in faces_keypoints]


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return list(np.linalg.norm(np.array(face_encodings) - np.array(face_to_compare), axis=1))


def main():
    img = cv2.imread("source.jpg")
    img1 = cv2.imread("img.jpg")
    img = img[:, :, ::-1]
    img1 = img1[:, :, ::-1]
    # 加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 加载人脸关键点检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 加载人脸检测编码模型
    encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 获取人脸128D编码
    img128d_features = get_128d_features(img, detector, predictor, encoder)
    img128d1_features = get_128d_features(img1, detector, predictor, encoder)

    if not img128d_features or not img128d1_features:
        print("No faces detected in one or both images.")
        return

    img128d = img128d_features[0]
    img128d1 = img128d1_features[0]

    # 计算人脸距离
    distance = face_distance(img128d, img128d1)
    print(distance)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
