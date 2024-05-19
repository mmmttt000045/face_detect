import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm

# 指定主文件夹路径
folder_path = 'imgs'

print('Running on device: {}'.format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize feature vector list
face_embeddings = []
person_names = []

# 遍历主文件夹中的每个子文件夹
for person_folder in tqdm(os.listdir(folder_path), desc="Processing Folders"):
    person_folder_path = os.path.join(folder_path, person_folder)

    # 检查是否是文件夹
    if os.path.isdir(person_folder_path):
        # 遍历子文件夹中的每个文件
        for file_name in os.listdir(person_folder_path):
            file_path = os.path.join(person_folder_path, file_name)

            # 检查文件是否是图片（可以根据需要添加更多的图片格式）
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 打开图片
                img = Image.open(file_path)
                img_cv2 = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                # 使用子文件夹名作为person_name
                person_name = person_folder

                # Detect faces
                boxes, _ = mtcnn.detect(img)

                # Extract feature vectors
                if boxes is not None:
                    # Get cropped face tensors
                    faces = mtcnn.extract(img, boxes, save_path=None)
                    if faces is not None:
                        # Move faces to the appropriate device
                        faces = faces.to(device)
                        # Get embeddings
                        embeddings = resnet(faces).detach().cpu().numpy()
                        face_embeddings.extend(embeddings)
                        person_names.extend([person_name] * len(embeddings))

                    # Draw bounding boxes on the image
                    for box in boxes:
                        cv2.rectangle(img_cv2,
                                      (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      (0, 255, 0), 2)

cv2.destroyAllWindows()

# 把person_names保存到一个txt文件中
if not os.path.exists('imgs_info'):
    os.makedirs('imgs_info')

with open('imgs_info/person_names.txt', 'w') as f:
    for name in person_names:
        f.write(name + '\n')

# 保存face_embeddings到npy文件
np.save('imgs_info/imgs_embeddings.npy', np.array(face_embeddings))

print('Extracted embeddings for {} faces.'.format(len(face_embeddings)))
for name in person_names:
    print(name)
