import sys
import time
import multiprocessing as mp
import cv2
import numpy as np

import config
import database
import face_utils

def webcam_capture_worker(frame_queue, stop_event, webcam_index=0):
    """
    Multiprocessing Worker (Process 1): Continuously captures frames from webcam
    and pushes the latest frame into a size-limited queue to prevent latency accumulation.
    """
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot access webcam index {webcam_index}.")
        frame_queue.put(None)
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Keep queue fresh: if full, discard oldest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Exception:
                pass

        try:
            frame_queue.put_nowait(frame)
        except Exception:
            pass

    cap.release()
    print("[INFO] Webcam capture process terminated.")

def register_student_flow():
    """Flow for registering a new student by capturing 100 face images."""
    print("\n--- STUDENT REGISTRATION ---")
    
    # Verify DB connection & student limit
    try:
        current_count = database.get_student_count()
    except Exception as e:
        print(f"[ERROR] Cannot connect to database: {e}")
        print("Please verify MySQL service is running and credentials in .env are correct.")
        return

    if current_count >= config.MAX_STUDENTS:
        print(f"[ERROR] Maximum student registration limit ({config.MAX_STUDENTS}) reached.")
        print("Registration cancelled.")
        return

    student_id = input("Enter Student ID: ").strip()
    if not student_id:
        print("[ERROR] Student ID cannot be empty.")
        return

    if database.get_student_by_id(student_id):
        print(f"[ERROR] Student ID '{student_id}' is already registered.")
        return

    name = input("Enter Student Full Name: ").strip()
    if not name:
        print("[ERROR] Student Name cannot be empty.")
        return

    classifier = face_utils.get_face_classifier()

    print(f"\nOpening webcam to capture {config.REGISTRATION_SAMPLE_COUNT} face samples for '{name}'...")
    print("Look at the camera. Press 'q' to cancel registration.\n")

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open webcam index {config.WEBCAM_INDEX}.")
        return

    captured_vectors = []
    
    while len(captured_vectors) < config.REGISTRATION_SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_utils.detect_faces(gray, classifier)

        for (x, y, w, h) in faces:
            # Draw bounding box for visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Preprocess face and extract HOG feature vector
            face_128 = face_utils.preprocess_face(gray, x, y, w, h)
            vec = face_utils.extract_hog_features(face_128)
            captured_vectors.append(vec)
            
            # Only process one face per frame for registration accuracy
            break

        count = len(captured_vectors)
        cv2.putText(
            frame, 
            f"Capturing: {count}/{config.REGISTRATION_SAMPLE_COUNT}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 255), 
            2
        )
        cv2.imshow("Student Registration - Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Registration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_vectors) < config.REGISTRATION_SAMPLE_COUNT:
        print("[ERROR] Incomplete capture. Registration aborted.")
        return

    print("\n[INFO] Processing captured faces...")
    # Calculate representative vector by averaging all 100 feature vectors
    mean_vector = np.mean(captured_vectors, axis=0)
    rep_vector = face_utils.normalize_vector(mean_vector)

    # Store in database
    success, msg = database.register_student(student_id, name, rep_vector)
    if success:
        print(f"[SUCCESS] {msg}")
    else:
        print(f"[ERROR] {msg}")

def start_attendance_flow():
    """
    Real-time face recognition attendance session using Multiprocessing.
    Process 1: Frame Capture
    Process 2 (Main): Processing, Cosine Matching, Attendance Update & GUI
    """
    print("\n--- STARTING ATTENDANCE SESSION ---")

    # Fetch registered students
    try:
        registered_students = database.get_all_students()
    except Exception as e:
        print(f"[ERROR] Could not load registered students from database: {e}")
        print("Please verify MySQL is running.")
        return

    if not registered_students:
        print("[WARNING] No students registered in database! Please register students first.")
        return

    # Initialize daily attendance records (Mark all registered as 'Absent' initially)
    added = database.initialize_daily_attendance()
    print(f"[INFO] Daily attendance initialized. {len(registered_students)} student records active for today.")

    classifier = face_utils.get_face_classifier()

    print("[INFO] Launching Multiprocessing pipeline...")
    frame_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    capture_process = mp.Process(
        target=webcam_capture_worker,
        args=(frame_queue, stop_event, config.WEBCAM_INDEX),
        daemon=True
    )
    capture_process.start()

    print("[INFO] Recognition running. Press 'q' in the window to stop attendance.\n")

    try:
        while True:
            if not capture_process.is_alive() and frame_queue.empty():
                print("[ERROR] Webcam capture process stopped unexpectedly.")
                break

            try:
                frame = frame_queue.get(timeout=2.0)
            except Exception:
                continue

            if frame is None:
                print("[ERROR] Received empty frame. Exiting webcam stream.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_utils.detect_faces(gray, classifier)

            for (x, y, w, h) in faces:
                face_128 = face_utils.preprocess_face(gray, x, y, w, h)
                query_vec = face_utils.extract_hog_features(face_128)
                query_vec_norm = face_utils.normalize_vector(query_vec)

                student_id, name, score = face_utils.find_best_match(
                    query_vec_norm,
                    registered_students,
                    threshold=config.SIMILARITY_THRESHOLD,
                    min_diff=config.MIN_MATCH_DIFF
                )

                if student_id is not None:
                    # Recognize registered student -> Mark Present
                    database.mark_student_present(student_id)
                    color = (0, 255, 0) # Green for recognized
                    label = f"{name} ({score:.2f})"
                else:
                    # Unknown person -> Do not create attendance
                    color = (0, 0, 255) # Red for unknown
                    label = f"Unknown ({score:.2f})"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # Display session info banner
            cv2.putText(
                frame,
                f"Attendance Active | Threshold: {config.SIMILARITY_THRESHOLD} | Press 'q' to Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.imshow("Real-Time Face Recognition Attendance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Attendance session ended by user.")
                break

    finally:
        # Proper termination and resource cleanup
        stop_event.set()
        capture_process.join(timeout=3.0)
        if capture_process.is_alive():
            capture_process.terminate()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Webcam and multiprocessing released.")

def delete_student_flow():
    """Flow for deleting a registered student by Student ID."""
    print("\n--- DELETE REGISTERED STUDENT ---")
    try:
        students = database.get_all_students()
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        return

    if not students:
        print("[INFO] No registered students found in database.")
        return

    print("Currently Registered Students:")
    for s in students:
        print(f"  • ID: {s['student_id']} | Name: {s['name']}")
    
    student_id = input("\nEnter Student ID to delete (or press Enter to cancel): ").strip()
    if not student_id:
        print("[INFO] Deletion cancelled.")
        return

    confirm = input(f"Are you sure you want to delete Student ID '{student_id}' and all their attendance history? (y/n): ").strip().lower()
    if confirm == 'y':
        success, msg = database.delete_student(student_id)
        if success:
            print(f"[SUCCESS] {msg}")
        else:
            print(f"[ERROR] {msg}")
    else:
        print("[INFO] Deletion cancelled.")

def view_attendance_flow():
    """Display attendance records for today and overall history."""
    print("\n--- ATTENDANCE RECORDS ---")
    try:
        records = database.fetch_attendance_report()
    except Exception as e:
        print(f"[ERROR] Could not fetch attendance records: {e}")
        print("Please check your MySQL database connection.")
        return

    if not records:
        print("No attendance records found.")
        return

    header = f"{'Student ID':<15} | {'Student Name':<20} | {'Status':<10} | {'Date':<12} | {'Time':<10}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    for rec in records:
        date_str = str(rec['attendance_date'])
        time_str = str(rec['attendance_time']) if rec['attendance_time'] else "N/A"
        status_colored = f"[PRESENT]" if rec['status'] == 'Present' else "[ABSENT]"
        print(f"{rec['student_id']:<15} | {rec['name']:<20} | {status_colored:<10} | {date_str:<12} | {time_str:<10}")
    print(divider)

def main_menu():
    """Terminal Menu Interface."""
    # Try initializing database schema on startup
    db_ok = database.initialize_database()

    while True:
        print("\n==============================================")
        print("  REAL-TIME FACE ATTENDANCE SYSTEM (HOG + MySQL)")
        print("==============================================")
        if not db_ok:
            print("[WARNING] MySQL database is offline or unconfigured.")
            print("Please ensure MySQL is running and set credentials in .env file.")
        print("1. Register Student (Max 3)")
        print("2. Start Attendance (Multiprocessing & Recognition)")
        print("3. View Attendance Records")
        print("4. Delete Registered Student")
        print("5. Exit")
        print("----------------------------------------------")

        choice = input("Select an option (1-5): ").strip()

        if choice == '1':
            register_student_flow()
        elif choice == '2':
            start_attendance_flow()
        elif choice == '3':
            view_attendance_flow()
        elif choice == '4':
            delete_student_flow()
        elif choice == '5':
            print("\nExiting Face Attendance System. Goodbye!")
            sys.exit(0)
        else:
            print("\n[INVALID SELECTION] Please choose a number between 1 and 5.")

if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility
    mp.freeze_support()
    main_menu()
