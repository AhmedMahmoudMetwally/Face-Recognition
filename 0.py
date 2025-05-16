import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, dataset_path='orl_faces'):
        """Initialize the data preprocessor with dataset path and parameters"""
        self.dataset_path = dataset_path
        self.image_size = (92, 112)  # Width x Height of each image
        self.num_subjects = 40  # Total number of subjects in dataset
        self.images_per_subject = 10  # Images per subject
        self.total_images = self.num_subjects * self.images_per_subject
        self.vector_length = self.image_size[0] * self.image_size[1]  # 92*112=10304
        
    def load_dataset(self):
        """Load dataset and convert images to numpy array"""
        # Initialize empty matrices for data and labels
        data_matrix = np.zeros((self.total_images, self.vector_length))
        label_vector = np.zeros(self.total_images)
        
        img_counter = 0
        # Loop through each subject folder (s1, s2, ..., s40)
        for subject_id in range(1, self.num_subjects + 1):
            subject_dir = os.path.join(self.dataset_path, f's{subject_id}')
            
            # Loop through each image for current subject (1.pgm to 10.pgm)
            for img_num in range(1, self.images_per_subject + 1):
                img_path = os.path.join(subject_dir, f'{img_num}.pgm')
                img = Image.open(img_path)
                
                # Convert image to 1D vector and store in data matrix
                img_vector = np.array(img).flatten()
                data_matrix[img_counter] = img_vector
                label_vector[img_counter] = subject_id
                
                img_counter += 1
                
        return data_matrix, label_vector
    
    def split_data(self, data_matrix, label_vector):
        """Split data into training and test sets"""
        # Use odd-indexed images for training (1st, 3rd, 5th, etc.)
        X_train = data_matrix[::2]
        # Use even-indexed images for testing (2nd, 4th, 6th, etc.)
        X_test = data_matrix[1::2]
        y_train = label_vector[::2]
        y_test = label_vector[1::2]
        
        return X_train, X_test, y_train, y_test
    
    def visualize_samples(self, data_matrix, label_vector, num_samples=5):
        """Display random samples from the dataset"""
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            # Select random image index
            idx = np.random.randint(0, len(data_matrix))
            # Reshape vector back to image dimensions
            img = data_matrix[idx].reshape(self.image_size)
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Subject {int(label_vector[idx])}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_pipeline(self):
        """Complete data preprocessing pipeline"""
        print("Loading dataset...")
        D, y = self.load_dataset()
        
        print("\nDisplaying sample images:")
        self.visualize_samples(D, y)
        
        print("\nSplitting dataset...")
        X_train, X_test, y_train, y_test = self.split_data(D, y)
        
        print("\nDataset information:")
        print(f"- Full data matrix shape: {D.shape}")
        print(f"- Training set shape: {X_train.shape}")
        print(f"- Test set shape: {X_test.shape}")
        print(f"- Number of classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    # Initialize preprocessor with dataset path
    preprocessor = DataPreprocessor(dataset_path=r"D:\Machine Learning\ASS 3 M\att_faces")
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Save processed data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)