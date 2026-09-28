import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "face_attendance")

# System Constraints & Thresholds
MAX_STUDENTS = int(os.getenv("MAX_STUDENTS", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.90))
MIN_MATCH_DIFF = float(os.getenv("MIN_MATCH_DIFF", 0.03))
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))

# Face Utility Settings
HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH", "haarcascade_frontalface_default.xml")
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/austinjoyal/haar-cascade-files/refs/heads/master/haarcascade_frontalface_default.xml"
FACE_IMAGE_SIZE = (128, 128)
REGISTRATION_SAMPLE_COUNT = 100
