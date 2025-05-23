import streamlit as st
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.linalg import eigh
import zipfile
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure the app
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("Face Recognition with ORL Dataset")

def safe_delete(path):
    """Helper function to delete files with retry logic"""
    try:
        if os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        time.sleep(0.1)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except:
            pass

def load_orl_dataset(zip_path):
    """Load ORL dataset with proper resource handling"""
    data_matrix = []
    labels = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for subject in range(1, 41):
            subject_dir = os.path.join(temp_dir, f"s{subject}")
            for img_num in range(1, 11):
                img_path = os.path.join(subject_dir, f"{img_num}.pgm")
                try:
                    with Image.open(img_path) as img:
                        img_array = np.array(img).flatten()
                        data_matrix.append(img_array)
                        labels.append(subject)
                    del img
                except FileNotFoundError:
                    continue
        
        gc.collect()
        
    finally:
        safe_delete(temp_dir)
    
    return np.array(data_matrix), np.array(labels)

def load_non_faces_dataset(num_images=50):
    """Improved non-faces images generation"""
    non_faces_data = []
    non_faces_labels = np.zeros(num_images)  # Label 0 for non-faces
    
    # Generate more realistic non-face patterns
    for _ in range(num_images):
        # Method 1: Random patterns with different distributions
        if np.random.rand() > 0.5:
            # Gaussian noise with different parameters
            random_img = np.random.normal(loc=100, scale=70, size=(112, 92))
        else:
            # Structured noise (stripes, blocks)
            if np.random.rand() > 0.5:
                # Vertical stripes
                random_img = np.zeros((112, 92))
                for i in range(0, 92, 10):
                    random_img[:, i:i+5] = np.random.randint(0, 256)
            else:
                # Blocks
                random_img = np.zeros((112, 92))
                for i in range(0, 112, 20):
                    for j in range(0, 92, 20):
                        random_img[i:i+10, j:j+10] = np.random.randint(0, 256)
        
        # Ensure values are in valid range
        random_img = np.clip(random_img, 0, 255).astype(np.uint8)
        non_faces_data.append(random_img.flatten())
    
    return np.array(non_faces_data), non_faces_labels

def pca(D, alpha=0.95):
    """PCA implementation with variance retention"""
    mu = np.mean(D, axis=0)
    Z = D - mu
    cov = np.cov(Z, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov)
    
    # Create writable copies
    eigenvalues = np.array(eigenvalues, copy=True)
    eigenvectors = np.array(eigenvectors, copy=True)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    total_variance = np.sum(eigenvalues)
    explained_variance = np.cumsum(eigenvalues)/total_variance
    r = np.argmax(explained_variance >= alpha) + 1
    Ur = eigenvectors[:, :r]
    return Ur, mu, explained_variance

def multiclass_lda(X, y, n_components=39):
    """Stable LDA implementation with regularization"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    mean_total = np.mean(X, axis=0)
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    
    for c in np.unique(y):
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        n_c = X_c.shape[0]
        Sw += (X_c - mean_c).T @ (X_c - mean_c)
        mean_diff = (mean_c - mean_total).reshape(-1, 1)
        Sb += n_c * (mean_diff @ mean_diff.T)
    
    Sw += np.eye(Sw.shape[0]) * 1e-6
    
    try:
        eigenvalues, eigenvectors = eigh(Sb, Sw)
    except:
        eigenvalues, eigenvectors = eigh(Sb, Sw + np.eye(Sw.shape[0]) * 1e-3)
    
    # Create writable copies
    eigenvalues = np.array(eigenvalues, copy=True)
    eigenvectors = np.array(eigenvectors, copy=True)
    
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx][:, :n_components].real
    return W

def main():
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Use relative path or let user upload the file
    dataset_path = st.sidebar.file_uploader("Upload ORL Dataset (ZIP file)", type=['zip'])
    
    if dataset_path is None:
        st.warning("Please upload the ORL dataset ZIP file to continue")
        return
    
    split_method = st.sidebar.radio("Train-Test Split", ["50-50 (Odd-Even)", "70-30"])
    pca_alpha = st.sidebar.slider("PCA Variance Retention (α)", 0.8, 0.95, 0.9, 0.05)
    k_value = st.sidebar.selectbox("k-NN Neighbors", [1, 3, 5, 7], index=0)
    
    # Save uploaded file temporarily
    temp_zip_path = os.path.join(tempfile.gettempdir(), "orl_dataset.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(dataset_path.getbuffer())
    
    # Load data
    with st.spinner("Loading dataset..."):
        try:
            data_matrix, labels = load_orl_dataset(temp_zip_path)
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")
            return
        finally:
            safe_delete(temp_zip_path)
    
    # Create splits
    if split_method == "50-50 (Odd-Even)":
        X_train = data_matrix[::2]
        X_test = data_matrix[1::2]
        y_train = labels[::2]
        y_test = labels[1::2]
    else:  # 70-30 split
        X_train, X_test, y_train, y_test = [], [], [], []
        for subject in np.unique(labels):
            subject_idx = np.where(labels == subject)[0]
            X_train.extend(data_matrix[subject_idx[:7]])
            X_test.extend(data_matrix[subject_idx[7:]])
            y_train.extend(labels[subject_idx[:7]])
            y_test.extend(labels[subject_idx[7:]])
        X_train, X_test = np.array(X_train), np.array(X_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["PCA Analysis", "LDA Analysis", "Performance Tuning", "Bonus: Face vs Non-Face"])
    
    with tab1:
        st.header("Principal Component Analysis")
        
        # Show accuracy for multiple alpha values
        st.subheader("PCA Performance Across Different α Values")
        alpha_values = [0.8, 0.85, 0.9, 0.95]
        results = []
        
        for alpha in alpha_values:
            Ur, mu, _ = pca(X_train, alpha)
            X_train_pca = (X_train - mu) @ Ur
            X_test_pca = (X_test - mu) @ Ur
            
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train_pca, y_train)
            y_pred = knn.predict(X_test_pca)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "α Value": alpha,
                "Components": Ur.shape[1],
                "Accuracy": acc
            })
        
        # Display results table
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.style.format({
            "Accuracy": "{:.2%}"
        }).highlight_max(subset=["Accuracy"], color='lightgreen'), 
        use_container_width=True)
        
        # Show detailed analysis for selected alpha
        st.subheader(f"Detailed Analysis for α = {pca_alpha}")
        with st.spinner("Running PCA..."):
            Ur, mu, explained_variance = pca(X_train, pca_alpha)
            X_train_pca = (X_train - mu) @ Ur
            X_test_pca = (X_test - mu) @ Ur
            
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train_pca, y_train)
            y_pred = knn.predict(X_test_pca)
            acc = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PCA Accuracy", f"{acc:.2%}")
                st.write(f"Components retained: {Ur.shape[1]}")
            
            with col2:
                eigenvalues = np.var(X_train_pca, axis=0)
                explained_variance = np.cumsum(eigenvalues)/np.sum(eigenvalues)
                
                fig, ax = plt.subplots()
                ax.plot(explained_variance)
                ax.axhline(y=pca_alpha, color='r', linestyle='--')
                ax.set_title('Explained Variance Ratio')
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Cumulative Explained Variance')
                st.pyplot(fig)
    
    with tab2:
        st.header("Linear Discriminant Analysis")
        st.subheader("Comparison Between PCA and LDA")
        
        with st.spinner("Running LDA..."):
            try:
                W = multiclass_lda(X_train, y_train)
                X_train_lda = X_train @ W
                X_test_lda = X_test @ W
                
                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train_lda, y_train)
                y_pred_lda = knn.predict(X_test_lda)
                lda_acc = accuracy_score(y_test, y_pred_lda)
                
                # Get PCA results for comparison
                Ur_pca, mu_pca, _ = pca(X_train, pca_alpha)
                X_train_pca = (X_train - mu_pca) @ Ur_pca
                X_test_pca = (X_test - mu_pca) @ Ur_pca
                knn_pca = KNeighborsClassifier(n_neighbors=k_value)
                knn_pca.fit(X_train_pca, y_train)
                y_pred_pca = knn_pca.predict(X_test_pca)
                pca_acc = accuracy_score(y_test, y_pred_pca)
                
                # Display comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("LDA Accuracy", f"{lda_acc:.2%}", 
                             delta=f"{(lda_acc-pca_acc):.2%} vs PCA")
                    st.write(f"Components used: {W.shape[1]}")
                
                with col2:
                    st.metric("PCA Accuracy", f"{pca_acc:.2%}", 
                             delta=f"{(pca_acc-lda_acc):.2%} vs LDA")
                    st.write(f"Components used: {Ur_pca.shape[1]}")
                
                # Simplified confusion matrix
                st.subheader("LDA Confusion Matrix (Sample Subjects)")
                cm = confusion_matrix(y_test, y_pred_lda)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm[:10,:10], annot=True, fmt='d', cmap='Blues', ax=ax)  # Show first 10 subjects
                ax.set_title('Confusion Matrix (Subjects 1-10)')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"LDA failed: {str(e)}")
                st.info("Trying PCA + LDA fallback...")
                
                Ur, mu = pca(X_train, 0.95)
                X_train_pca = (X_train - mu) @ Ur
                X_test_pca = (X_test - mu) @ Ur
                
                W = multiclass_lda(X_train_pca, y_train)
                X_train_lda = X_train_pca @ W
                X_test_lda = X_test_pca @ W
                
                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train_lda, y_train)
                y_pred_lda = knn.predict(X_test_lda)
                lda_acc = accuracy_score(y_test, y_pred_lda)
                
                st.metric("LDA on PCA-Reduced Data Accuracy", f"{lda_acc:.2%}")
    
    with tab3:
        st.header("Performance Tuning")
        with st.spinner("Evaluating different configurations..."):
            k_values = [1, 3, 5, 7]
            pca_accs = []
            lda_accs = []
            
            for k in k_values:
                # PCA evaluation
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_pca, y_train)
                pca_accs.append(accuracy_score(y_test, knn.predict(X_test_pca)))
                
                # LDA evaluation
                if 'W' in locals():
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_lda, y_train)
                    lda_accs.append(accuracy_score(y_test, knn.predict(X_test_lda)))
            
            fig, ax = plt.subplots()
            ax.plot(k_values, pca_accs, 'bo-', label='PCA')
            if lda_accs:
                ax.plot(k_values, lda_accs, 'ro-', label='LDA')
            ax.set_title('k-NN Performance by Neighbors')
            ax.set_xlabel('Number of Neighbors (k)')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Split Method Comparison")
            st.write(f"Current split: {split_method}")
            st.write(f"Training samples per subject: {7 if split_method == '70-30' else 5}")
            st.write(f"Testing samples per subject: {3 if split_method == '70-30' else 5}")
    
    with tab4:
        st.header("Face vs Non-Face Classification")
        st.write("Binary classification to distinguish between face and non-face images.")
        
        num_non_faces = st.slider("Number of Non-Face Images", 50, 400, 200, 50)
        
        with st.spinner("Loading non-faces images..."):
            non_faces_data, non_faces_labels = load_non_faces_dataset(num_non_faces)
        
        # Combine faces and non-faces data
        X_train_combined = np.vstack([X_train, non_faces_data[:num_non_faces//2]])
        X_test_combined = np.vstack([X_test, non_faces_data[num_non_faces//2:]])
        y_train_combined = np.hstack([y_train, non_faces_labels[:num_non_faces//2]])
        y_test_combined = np.hstack([y_test, non_faces_labels[num_non_faces//2:]])
        
        # Binary classification (face vs non-face)
        y_train_binary = (y_train_combined > 0).astype(int)
        y_test_binary = (y_test_combined > 0).astype(int)
        
        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = scaler.transform(X_test_combined)
        
        st.subheader("PCA Approach")
        Ur_combined, mu_combined, _ = pca(X_train_scaled, 0.95)
        X_train_pca_combined = (X_train_scaled - mu_combined) @ Ur_combined
        X_test_pca_combined = (X_test_scaled - mu_combined) @ Ur_combined
        
        # Improved KNN model
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine'
        )
        knn.fit(X_train_pca_combined, y_train_binary)
        y_pred_binary = knn.predict(X_test_pca_combined)
        acc_binary = accuracy_score(y_test_binary, y_pred_binary)
        
        # Enhanced binary classification report
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{acc_binary:.2%}")
            
            # Classification report
            report = classification_report(y_test_binary, y_pred_binary, 
                                         target_names=["Non-Face", "Face"], output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}"
            }).highlight_max(subset=["precision", "recall", "f1-score"], color='lightgreen'), 
            use_container_width=True)
        
        with col2:
            # Confusion matrix
            cm = confusion_matrix(y_test_binary, y_pred_binary)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Non-Face", "Face"],
                        yticklabels=["Non-Face", "Face"])
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        # Show examples with improved visualization
        st.subheader("Sample Classifications")
        st.write("Random examples from test set with predictions:")
        
        # Get misclassified examples first
        wrong_indices = np.where(y_pred_binary != y_test_binary)[0]
        correct_indices = np.where(y_pred_binary == y_test_binary)[0]
        
        # Show 4 wrong and 4 correct predictions
        sample_indices = (list(np.random.choice(wrong_indices, size=min(4, len(wrong_indices)), replace=False)) + 
                         list(np.random.choice(correct_indices, size=8-min(4, len(wrong_indices)), replace=False)))
        
        cols = st.columns(4)
        
        for i, idx in enumerate(sample_indices):
            img = X_test_combined[idx].reshape(112, 92)
            pred = "Face" if y_pred_binary[idx] else "Non-Face"
            true = "Face" if y_test_binary[idx] else "Non-Face"
            color = "green" if pred == true else "red"
            
            with cols[i % 4]:
                st.image(img, use_column_width=True)
                st.markdown(f"**Prediction:** <span style='color:{color}'>{pred}</span>", 
                           unsafe_allow_html=True)
                st.caption(f"Actual: {true}")
                if pred != true:
                    st.error("Misclassified")

if __name__ == "__main__":
    main()