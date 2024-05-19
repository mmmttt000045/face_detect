import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

import utils.util

# 检测设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(torch.cuda.get_device_name(0)))

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load previously saved feature vectors
saved_embeddings = np.load('imgs_info/npy/imgs_embeddings.npy')
print(saved_embeddings.shape)
camera = cv2.VideoCapture(0)
# Load names
names = utils.util.read_txt('imgs_info/person_names.txt')
print(names)
try:
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break

        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            # Extract face tensors
            faces = mtcnn.extract(frame, boxes, save_path=None)
            if faces is not None:
                # Move faces to the correct device
                faces = faces.to(device)
                # Get embeddings
                new_embeddings = resnet(faces).detach().cpu().numpy()

                # Compare each detected face with saved embeddings
                for i, new_embedding in enumerate(new_embeddings):
                    # Calculate cosine distance
                    distances = [cosine(new_embedding, emb) for emb in saved_embeddings]
                    min_distance = min(distances)
                    min_index = distances.index(min_distance)

                    # Set a threshold and display match results
                    if min_distance < 0.3:
                        match_message = f"{names[min_index]}  Distance:{min_distance:.2f} "
                    else:
                        match_message = "No match found"

                    # Draw bounding boxes and match results on the frame
                    box = boxes[i]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, match_message, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Face Detection and Recognition', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
