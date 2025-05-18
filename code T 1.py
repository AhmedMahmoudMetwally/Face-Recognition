import os
import numpy as np
import cv2

# 1. Load all face images from the dataset folder
def load_faces_dataset(dataset_path=r"D:\Machine Learning\ASS 3 M\att_faces"):
    data = []
    labels = []

    for subject_id in range(1, 41):  # 40 subjects
        folder_path = os.path.join(dataset_path, f"s{subject_id}")
        images = sorted(os.listdir(folder_path))  # Ensure images are in order 1-10

        for idx, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                data.append(img.flatten())  # Flatten the image: shape (10304,)
                labels.append(subject_id)
            else:
                print(f"Could not read image: {img_path}")

    data_matrix = np.array(data)      # Final shape: (400, 10304)
    label_vector = np.array(labels)   # Final shape: (400,)
    return data_matrix, label_vector

# 2. Load the dataset once
X, y = load_faces_dataset(r"D:\Machine Learning\ASS 3 M\att_faces")

# 3. Split the dataset: odd-indexed images for training, even-indexed for testing
X_train, y_train = X[::2], y[::2]
X_test, y_test = X[1::2], y[1::2]

# 4. Print shapes of the datasets
print("X shape      :", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# 5. Save the data to disk (optional)
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
