import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
import matplotlib.pyplot as plt

def generate_sample_camera_data(n_samples=1000, n_features=10, anomaly_ratio=0.1):
    """
    Generate sample camera data for anomaly detection.
    In a real scenario, this would be replaced with actual camera data processing.
    
    Features might include:
    - Image brightness
    - Image contrast
    - Edge density
    - Color distribution
    - Motion level
    - Blur level
    - Noise level
    - Object count
    - Frame difference
    - Compression artifacts
    """
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_ratio))
    normal_data = np.random.normal(0, 1, (n_normal, n_features))
    
    # Generate anomalous data (shifted distribution)
    n_anomalies = n_samples - n_normal
    anomaly_data = np.random.normal(3, 2, (n_anomalies, n_features))
    
    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    
    # Create labels (0 for normal, 1 for anomaly)
    labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    return data[indices], labels[indices]

def extract_image_features(image):
    """
    Extract features from an image that can be used for anomaly detection.
    In a real implementation, you would replace this with actual feature extraction.
    """
    features = []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Feature 1: Average brightness
    features.append(np.mean(gray))
    
    # Feature 2: Standard deviation (contrast)
    features.append(np.std(gray))
    
    # Feature 3: Edge density
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]))
    
    # Feature 4: Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(laplacian_var)
    
    # Feature 5: Histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    features.append(np.mean(hist))
    features.append(np.std(hist))
    
    # Feature 6: Color distribution (if color image)
    if len(image.shape) == 3:
        for i in range(3):
            features.append(np.mean(image[:, :, i]))
            features.append(np.std(image[:, :, i]))
    else:
        # For grayscale, duplicate values
        for i in range(6):
            features.append(features[0])
    
    return np.array(features)

def simulate_camera_stream(n_frames=200):
    """
    Simulate a camera stream with some anomalies.
    In a real implementation, this would connect to an actual camera.
    """
    print("Simulating camera stream...")
    
    # Generate base normal data
    normal_frames = []
    for i in range(int(n_frames * 0.9)):
        # Create a normal frame (mostly uniform with some noise)
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # Add some structured elements
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
        normal_frames.append(frame)
    
    # Generate anomalous frames
    anomaly_frames = []
    for i in range(int(n_frames * 0.1)):
        # Create an anomalous frame (different characteristics)
        frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        # Add different structured elements
        cv2.circle(frame, (300, 300), 50, (0, 0, 255), 3)
        anomaly_frames.append(frame)
    
    # Combine and shuffle
    all_frames = normal_frames + anomaly_frames
    np.random.shuffle(all_frames)
    
    return all_frames

def main():
    print("Camera Anomaly Detection with PyOD")
    print("=" * 40)
    
    # Option 1: Using generated sample data
    print("\n1. Using generated sample data:")
    X, y_true = generate_sample_camera_data(n_samples=1000, n_features=10, anomaly_ratio=0.1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Isolation Forest': IForest(contamination=0.1, random_state=42),
        'One-Class SVM': OCSVM(contamination=0.1),
        'KNN': KNN(contamination=0.1),
        'CBLOF': CBLOF(contamination=0.1, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        # Fit the model
        model.fit(X_train_scaled)
        
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        y_scores = model.decision_scores_
        
        # Store results
        results[name] = {
            'predictions': y_pred,
            'scores': y_scores
        }
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"{name} Accuracy: {accuracy:.3f}")
    
    # Option 2: Simulate camera stream processing
    print("\n2. Simulating camera stream processing:")
    frames = simulate_camera_stream(n_frames=100)
    
    # Extract features from frames
    features = []
    for i, frame in enumerate(frames[:20]):  # Process first 20 frames for demo
        frame_features = extract_image_features(frame)
        features.append(frame_features)
    
    features = np.array(features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply anomaly detection
    model = IForest(contamination=0.1, random_state=42)
    model.fit(features_scaled)
    
    # Predict anomalies
    predictions = model.predict(features_scaled)
    anomaly_scores = model.decision_scores_
    
    print(f"Processed {len(frames[:20])} frames")
    print(f"Detected {np.sum(predictions)} anomalies")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Sample data results
    plt.subplot(2, 2, 1)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm')
    plt.title('True Labels (Sample Data)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 2, 2)
    best_model_name = 'Isolation Forest'  # Just pick one for visualization
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=results[best_model_name]['predictions'], cmap='coolwarm')
    plt.title(f'{best_model_name} Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 3: Anomaly scores distribution
    plt.subplot(2, 2, 3)
    plt.hist(anomaly_scores, bins=30, alpha=0.7)
    plt.title('Anomaly Scores Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    
    # Plot 4: Camera stream results
    plt.subplot(2, 2, 4)
    plt.plot(anomaly_scores, 'o-')
    plt.title('Anomaly Scores Over Time (Camera Stream)')
    plt.xlabel('Frame')
    plt.ylabel('Anomaly Score')
    plt.axhline(y=np.mean(anomaly_scores) + 2*np.std(anomaly_scores), color='r', linestyle='--', 
                label='Anomaly Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('camera_anomaly_detection_results.png')
    plt.show()
    
    print("\n3. How to use with real camera:")
    print("""
To use with a real camera, replace the simulate_camera_stream function with:

import cv2

def capture_camera_stream():
    cap = cv2.VideoCapture(0)  # Use default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    model = IForest(contamination=0.1)
    # You would need to train this model with normal data first
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract features from frame
        features = extract_image_features(frame)
        
        # Predict if anomaly (you would need to scale features and use a trained model)
        # prediction = model.predict([features])
        
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    """)

if __name__ == "__main__":
    main()