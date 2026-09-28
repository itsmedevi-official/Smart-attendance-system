-- Schema for Face Recognition Attendance System

CREATE DATABASE IF NOT EXISTS face_attendance;
USE face_attendance;

-- Table to store student details and their HOG feature vectors
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    face_vector LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store attendance records per student per day
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL,
    status ENUM('Present', 'Absent') DEFAULT 'Absent',
    attendance_date DATE NOT NULL,
    attendance_time TIME NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    CONSTRAINT unique_student_date UNIQUE (student_id, attendance_date)
);
