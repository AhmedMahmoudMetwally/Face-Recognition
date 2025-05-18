import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import zipfile
import tempfile
import shutil

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
            
            # Check if subject directory exists
            if not os.path.exists(subject_dir):
                raise FileNotFoundError(f"Directory not found: {subject_dir}")
            
            # Loop through each image for current subject (1.pgm to 10.pgm)
            for img_num in range(1, self.images_per_subject + 1):
                img_path = os.path.join(subject_dir, f'{img_num}.pgm')
                
                # Check if image file exists
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                
                try:
                    with Image.open(img_path) as img:
                        # Convert image to 1D vector and store in data matrix
                        img_vector = np.array(img).flatten()
                        data_matrix[img_counter] = img_vector
                        label_vector[img_counter] = subject_id
                        
                    img_counter += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue
                
        return data_matrix[:img_counter], label_vector[:img_counter]
    
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
    
    def run_pipeline(self):
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

class FaceRecognizer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pca_results = {}
        self.lda_results = None
    
    def pca(self, alpha=0.95):
        """Perform PCA with specified variance retention level"""
        # Center the data
        mu = np.mean(self.X_train, axis=0)
        Z = self.X_train - mu
        
        # Compute covariance matrix
        cov = np.cov(Z, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(cov)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components to retain
        total_variance = np.sum(eigenvalues)
        explained_variance = np.cumsum(eigenvalues)/total_variance
        r = np.argmax(explained_variance >= alpha) + 1
        
        # Select top r components
        Ur = eigenvectors[:, :r]
        
        # Project data
        X_train_pca = (self.X_train - mu) @ Ur
        X_test_pca = (self.X_test - mu) @ Ur
        
        return X_train_pca, X_test_pca, Ur, mu
    
    def evaluate_pca(self, alphas=[0.80, 0.85, 0.90, 0.95]):
        """Evaluate PCA with different variance retention levels"""
        results = {}
        
        for alpha in alphas:
            # Perform PCA
            X_train_pca, X_test_pca, _, _ = self.pca(alpha)
            
            # Train 1-NN classifier
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_pca, self.y_train)
            
            # Predict and evaluate
            y_pred = knn.predict(X_test_pca)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[alpha] = {
                'accuracy': accuracy,
                'n_components': X_train_pca.shape[1]
            }
        
        self.pca_results = results
        return results
    
    def lda(self):
        """Perform LDA for multiclass classification"""
        n_samples, n_features = self.X_train.shape
        n_classes = len(np.unique(self.y_train))
        
        # Compute overall mean
        mean_total = np.mean(self.X_train, axis=0)
        
        # Initialize scatter matrices
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        # Compute within-class and between-class scatter
        for c in np.unique(self.y_train):
            X_c = self.X_train[self.y_train == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]
            
            # Within-class scatter
            Sw += (X_c - mean_c).T @ (X_c - mean_c)
            
            # Between-class scatter
            mean_diff = (mean_c - mean_total).reshape(-1, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)
        
        # Add regularization for numerical stability
        Sw += np.eye(Sw.shape[0]) * 1e-6
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(Sb, Sw)
        
        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        W = eigenvectors[:, idx][:, :n_classes-1].real  # Take top (n_classes-1) components
        
        # Project data
        X_train_lda = self.X_train @ W
        X_test_lda = self.X_test @ W
        
        return X_train_lda, X_test_lda, W
    
    def evaluate_lda(self):
        """Evaluate LDA performance"""
        # Perform LDA
        X_train_lda, X_test_lda, _ = self.lda()
        
        # Train 1-NN classifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_lda, self.y_train)
        
        # Predict and evaluate
        y_pred = knn.predict(X_test_lda)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.lda_results = accuracy
        return accuracy
    
    def visualize_results(self):
        """Visualize PCA and LDA results"""
        # PCA results visualization
        plt.figure(figsize=(15, 5))
        
        # PCA accuracy vs alpha
        plt.subplot(1, 2, 1)
        alphas = list(self.pca_results.keys())
        accuracies = [self.pca_results[a]['accuracy'] for a in alphas]
        n_components = [self.pca_results[a]['n_components'] for a in alphas]
        
        plt.plot(alphas, accuracies, 'bo-')
        plt.title('PCA Classification Accuracy vs Variance Retention')
        plt.xlabel('Variance Retention (α)')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        for i, txt in enumerate(n_components):
            plt.annotate(f'n={txt}', (alphas[i], accuracies[i]))
        
        # LDA result
        plt.subplot(1, 2, 2)
        plt.bar(['LDA'], [self.lda_results])
        plt.title('LDA Classification Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison
        print("\nComparison between PCA and LDA:")
        print(f"Best PCA Accuracy (α=0.95): {self.pca_results[0.95]['accuracy']:.2%}")
        print(f"LDA Accuracy: {self.lda_results:.2%}")

def find_dataset_root(extracted_path):
    """Recursively search for the folder containing s1, s2,... subfolders"""
    for root, dirs, files in os.walk(extracted_path):
        if any(f's{i}' in dirs for i in range(1, 41)):
            return root
    return None

def extract_dataset(zip_path, extract_to):
    """Extract ORL dataset from zip file and return correct path"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Search for the actual dataset folder
    dataset_root = find_dataset_root(extract_to)
    
    if dataset_root is None:
        # Check if the extracted folder itself contains the dataset
        if os.path.exists(os.path.join(extract_to, 's1')):
            return extract_to
        raise FileNotFoundError("Could not find dataset folders (s1, s2,...) in extracted files")
    
    return dataset_root

# Main execution
if __name__ == "__main__":
    # Extract dataset
    zip_path = r"C:\Users\user\Downloads\archive (6).zip"
    extract_to = tempfile.mkdtemp()
    
    try:
        print(f"Extracting dataset from {zip_path}...")
        dataset_path = extract_dataset(zip_path, extract_to)
        print(f"Found dataset at: {dataset_path}")
        
        # Initialize preprocessor with dataset path
        preprocessor = DataPreprocessor(dataset_path=dataset_path)
        
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
        
        # Initialize face recognizer
        recognizer = FaceRecognizer(X_train, X_test, y_train, y_test)
        
        # Evaluate PCA with different alphas
        print("\nEvaluating PCA with different variance retention levels...")
        pca_results = recognizer.evaluate_pca()
        print("\nPCA Results:")
        for alpha, res in pca_results.items():
            print(f"α={alpha:.2f}: Accuracy={res['accuracy']:.2%}, Components={res['n_components']}")
        
        # Evaluate LDA
        print("\nEvaluating LDA...")
        lda_accuracy = recognizer.evaluate_lda()
        print(f"LDA Accuracy: {lda_accuracy:.2%}")
        
        # Visualize results
        recognizer.visualize_results()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check:")
        print("1. The zip file path is correct")
        print("2. The zip file contains the ORL dataset folders (s1, s2, etc.)")
        print("3. The folder structure inside the zip file")
        print("You can manually inspect the zip file contents to verify")
    finally:
        # Clean up temporary files
        shutil.rmtree(extract_to, ignore_errors=True)