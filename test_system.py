import unittest
import json
import numpy as np
import cv2

import config
import face_utils

class TestFaceAttendanceSystem(unittest.TestCase):

    def test_haar_cascade_file_exists(self):
        """Test that Haar cascade XML exists and classifier loads without errors."""
        face_utils.ensure_haar_cascade_file()
        classifier = face_utils.get_face_classifier()
        self.assertFalse(classifier.empty(), "Haar Cascade classifier should not be empty")

    def test_hog_feature_extraction(self):
        """Test HOG feature extraction on a synthetic 128x128 grayscale image."""
        synthetic_face = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        equalized = cv2.equalizeHist(synthetic_face)
        features = face_utils.extract_hog_features(equalized)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        print(f"[TEST PASS] HOG feature vector length: {len(features)}")

    def test_vector_normalization(self):
        """Test vector L2 normalization."""
        vec = np.array([3.0, 4.0], dtype=np.float64)
        norm_vec = face_utils.normalize_vector(vec)
        self.assertAlmostEqual(np.linalg.norm(norm_vec), 1.0, places=5)

    def test_cosine_similarity_exact_and_orthogonal(self):
        """Test cosine similarity for identical and orthogonal vectors."""
        vec_a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        vec_b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        vec_c = np.array([-2.0, 1.0, 0.0], dtype=np.float64) # Orthogonal to vec_a in terms of dot product: -2 + 2 + 0 = 0

        sim_identical = face_utils.compute_cosine_similarity(vec_a, vec_b)
        sim_orthogonal = face_utils.compute_cosine_similarity(vec_a, vec_c)

        self.assertAlmostEqual(sim_identical, 1.0, places=5)
        self.assertAlmostEqual(sim_orthogonal, 0.0, places=5)

    def test_best_match_matching_and_unknown(self):
        """Test best match matching threshold and unknown person categorization."""
        vec_student1 = face_utils.normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        vec_student2 = face_utils.normalize_vector(np.array([0.0, 1.0, 0.0], dtype=np.float64))

        registered_students = [
            {'student_id': 'S101', 'name': 'Alice', 'vector': vec_student1},
            {'student_id': 'S102', 'name': 'Bob', 'vector': vec_student2},
        ]

        # Case 1: Query identical to Alice -> should match Alice
        query_alice = face_utils.normalize_vector(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        s_id, name, score = face_utils.find_best_match(query_alice, registered_students, threshold=0.90, min_diff=0.03)
        self.assertEqual(s_id, 'S101')
        self.assertEqual(name, 'Alice')
        self.assertGreaterEqual(score, 0.90)

        # Case 2: Query low similarity to all -> should return Unknown Person
        query_unknown = face_utils.normalize_vector(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        s_id, name, score = face_utils.find_best_match(query_unknown, registered_students, threshold=0.90, min_diff=0.03)
        self.assertIsNone(s_id)
        self.assertEqual(name, 'Unknown Person')

    def test_vector_json_serialization(self):
        """Test that feature vectors serialize and deserialize from JSON correctly without precision loss."""
        vec = np.random.rand(1764).astype(np.float64)
        json_str = json.dumps(vec.tolist())
        reconstructed = np.array(json.loads(json_str), dtype=np.float64)
        
        np.testing.assert_allclose(vec, reconstructed, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
