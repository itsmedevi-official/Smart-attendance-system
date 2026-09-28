import os
import urllib.request
import cv2
import numpy as np
from skimage.feature import hog

import config

def ensure_haar_cascade_file():
    """Verify Haar Cascade XML file exists; download if missing."""
    filepath = config.HAAR_CASCADE_PATH
    if not os.path.exists(filepath):
        print(f"[INFO] Haar Cascade file not found at '{filepath}'. Downloading...")
        try:
            urllib.request.urlretrieve(config.HAAR_CASCADE_URL, filepath)
            print("[INFO] Haar Cascade downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Haar Cascade file: {e}")

def get_face_classifier():
    """Load OpenCV Haar Cascade face classifier."""
    ensure_haar_cascade_file()
    classifier = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
    if classifier.empty():
        raise ValueError(f"Failed to load Haar Cascade XML from {config.HAAR_CASCADE_PATH}")
    return classifier

def detect_faces(gray_img, classifier):
    """Detect faces in a grayscale image frame using Haar Cascade."""
    faces = classifier.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    return faces

def preprocess_face(gray_img, x, y, w, h, target_size=config.FACE_IMAGE_SIZE):
    """
    Extract face ROI, apply histogram equalization for contrast normalization,
    and resize to standard dimensions (128x128).
    """
    face_roi = gray_img[y:y+h, x:x+w]
    equalized_face = cv2.equalizeHist(face_roi)
    resized_face = cv2.resize(equalized_face, target_size, interpolation=cv2.INTER_AREA)
    return resized_face

def extract_hog_features(face_128x128):
    """
    Extract HOG (Histogram of Oriented Gradients) feature vector from 128x128 face region.
    Default config: 9 orientations, 16x16 pixels per cell, 2x2 cells per block.
    """
    features = hog(
        face_128x128,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True
    )
    return features.astype(np.float64)

def normalize_vector(vec):
    """Normalize feature vector to unit length (L2 norm)."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two feature vectors:
    cosine_similarity = dot(A, B) / (norm(A) * norm(B))
    """
    vec_a = np.array(vec_a, dtype=np.float64)
    vec_b = np.array(vec_b, dtype=np.float64)
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    dot_product = np.dot(vec_a, vec_b)
    similarity = dot_product / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))

def find_best_match(query_vec, registered_students, threshold=config.SIMILARITY_THRESHOLD, min_diff=config.MIN_MATCH_DIFF):
    """
    Compare query vector against all registered student vectors.
    Returns (student_id, student_name, similarity_score).
    If no match exceeds threshold or match difference is too small, returns (None, "Unknown Person", score).
    """
    if not registered_students:
        return None, "Unknown Person", 0.0

    scores = []
    for student in registered_students:
        sim = compute_cosine_similarity(query_vec, student['vector'])
        scores.append((sim, student['student_id'], student['name']))

    # Sort descending by similarity score
    scores.sort(key=lambda item: item[0], reverse=True)

    best_score, best_id, best_name = scores[0]

    # Check primary similarity threshold
    if best_score < threshold:
        return None, "Unknown Person", best_score

    # Check minimum difference from second best match if more than 1 student is registered
    if len(scores) > 1:
        second_score = scores[1][0]
        diff = best_score - second_score
        if diff < min_diff:
            # Ambiguous match (two registered students have almost identical similarity)
            return None, "Unknown Person", best_score

    return best_id, best_name, best_score
