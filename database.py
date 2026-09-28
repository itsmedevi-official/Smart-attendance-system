import json
from datetime import date
import mysql.connector
from mysql.connector import Error
import numpy as np

import config

def get_db_connection(use_db=True):
    """Establish connection to MySQL server."""
    try:
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME if use_db else None
        )
        return connection
    except Error as e:
        # Caller will catch and handle database connection errors
        raise ConnectionError(f"Database connection failed: {e}")

def initialize_database():
    """Create database and tables if they do not exist."""
    try:
        # Step 1: Connect without DB to ensure DB exists
        conn = get_db_connection(use_db=False)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME}")
        cursor.close()
        conn.close()

        # Step 2: Connect with DB and create tables
        conn = get_db_connection(use_db=True)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            face_vector LONGTEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(50) NOT NULL,
            status ENUM('Present', 'Absent') DEFAULT 'Absent',
            attendance_date DATE NOT NULL,
            attendance_time TIME NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            CONSTRAINT unique_student_date UNIQUE (student_id, attendance_date)
        );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Database initialization verified successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        return False

def get_student_count():
    """Return total number of registered students."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count

def get_student_by_id(student_id):
    """Retrieve student details by ID."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT student_id, name FROM students WHERE student_id = %s", (student_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

def delete_student(student_id):
    """Delete a student and their attendance history by Student ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "DELETE FROM students WHERE student_id = %s"
        cursor.execute(query, (student_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            return True, f"Student ID '{student_id}' and associated attendance records deleted successfully."
        else:
            return False, f"Student ID '{student_id}' not found in database."
    except Error as e:
        conn.rollback()
        return False, f"Failed to delete student: {e}"
    finally:
        cursor.close()
        conn.close()

def register_student(student_id, name, feature_vector):
    """
    Register a new student with representative HOG feature vector serialized as JSON.
    Enforces maximum student limit and unique student ID.
    """
    current_count = get_student_count()
    if current_count >= config.MAX_STUDENTS:
        return False, f"Maximum student limit ({config.MAX_STUDENTS}) reached. Cannot register more students."

    if get_student_by_id(student_id):
        return False, f"Student ID '{student_id}' already exists."

    vector_json = json.dumps(feature_vector.tolist())

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO students (student_id, name, face_vector) VALUES (%s, %s, %s)"
        cursor.execute(query, (student_id, name, vector_json))
        conn.commit()
        return True, f"Student '{name}' (ID: {student_id}) registered successfully!"
    except Error as e:
        conn.rollback()
        return False, f"Failed to register student: {e}"
    finally:
        cursor.close()
        conn.close()

def get_all_students():
    """Retrieve all registered students and deserialize their HOG feature vectors."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT student_id, name, face_vector FROM students")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    students = []
    for row in rows:
        vec_list = json.loads(row['face_vector'])
        vec_np = np.array(vec_list, dtype=np.float64)
        students.append({
            'student_id': row['student_id'],
            'name': row['name'],
            'vector': vec_np
        })
    return students

def initialize_daily_attendance():
    """
    Create 'Absent' attendance records for all registered students for today if not present.
    Prevents duplicate entries via UNIQUE(student_id, attendance_date).
    """
    students = get_all_students()
    if not students:
        return 0

    today = date.today()
    conn = get_db_connection()
    cursor = conn.cursor()
    added_count = 0

    try:
        for student in students:
            query = """
                INSERT IGNORE INTO attendance (student_id, status, attendance_date)
                VALUES (%s, 'Absent', %s)
            """
            cursor.execute(query, (student['student_id'], today))
            if cursor.rowcount > 0:
                added_count += 1
        conn.commit()
    except Error as e:
        conn.rollback()
        print(f"[ERROR] Failed to initialize daily attendance: {e}")
    finally:
        cursor.close()
        conn.close()

    return added_count

def mark_student_present(student_id):
    """
    Mark student as Present for today and record recognition time.
    Only updates if status is currently 'Absent' to preserve initial timestamp.
    """
    today = date.today()
    conn = get_db_connection()
    cursor = conn.cursor()
    updated = False

    try:
        query = """
            UPDATE attendance 
            SET status = 'Present', attendance_time = CURRENT_TIME()
            WHERE student_id = %s AND attendance_date = %s AND status = 'Absent'
        """
        cursor.execute(query, (student_id, today))
        conn.commit()
        if cursor.rowcount > 0:
            updated = True
    except Error as e:
        conn.rollback()
        print(f"[ERROR] Failed to mark attendance for {student_id}: {e}")
    finally:
        cursor.close()
        conn.close()

    return updated

def fetch_attendance_report():
    """Retrieve all attendance records with student names."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT a.attendance_id, a.student_id, s.name, a.status, a.attendance_date, a.attendance_time
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.attendance_date DESC, a.attendance_time ASC, s.name ASC
    """
    cursor.execute(query)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records
