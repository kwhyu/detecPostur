import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, StringVar, messagebox, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk
import sqlite3
import json
import os
import numpy as np
from time import time
import atexit
import logging
import threading


# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger()

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables
recording = False
recorded_data = []
last_frame_time = 0
frame_interval = 0.03  # 10ms interval (~100 FPS)

# Database setup
DB_FILE = "dance.db"

def initialize_database():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS motions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dance_id INTEGER NOT NULL,
                landmarks TEXT NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (dance_id) REFERENCES dances (id)
            )
        """)
        conn.commit()
    logger.info("Database initialized.")

def add_dance(name):
    """Add a dance type to the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO dances (name) VALUES (?)", (name,))
        conn.commit()
    logger.debug(f"Dance '{name}' added to database.")

def get_all_dances():
    """Get all dance types from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM dances")
        return cursor.fetchall()

def add_motion(dance_id, landmarks, score):
    """Add motion data to the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO motions (dance_id, landmarks, score)
            VALUES (?, ?, ?)
        """, (dance_id, json.dumps(landmarks), score))
        conn.commit()
    logger.debug(f"Motion data saved for dance ID {dance_id} with score {score}.")

def get_dance_motions(dance_id):
    """Get all motions for a specific dance."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT landmarks, score FROM motions WHERE dance_id = ?", (dance_id,))
        return cursor.fetchall()

def calculate_score(reference, current):
    if len(reference) != len(current):
        return 0

    threshold = 0.05
    differences = [
        max(0, np.linalg.norm(np.array(r) - np.array(c)) - threshold)
        for r, c in zip(reference, current)
    ]

    average_difference = sum(differences) / len(differences)
    score = max(0, 100 - int(average_difference * 50))
    logger.debug(f"Thresholded Differences: {differences}, Average: {average_difference}, Score: {score}")
    return score

def evaluate_motion(dance_id, recorded_data):
    """Evaluate the recorded motion against a dance type."""
    motions = get_dance_motions(dance_id)
    if not motions or not recorded_data:
        logger.warning("No data available for evaluation.")
        return "Tidak ada data untuk penilaian."

    best_score = 0
    for motion in motions:
        reference_landmarks = json.loads(motion[0])
        score = calculate_score(reference_landmarks, recorded_data)
        best_score = max(best_score, score)

    logger.info(f"Best score: {best_score}")
    return f"Skor terbaik: {best_score}"

def filter_landmarks(landmarks):
    """Take all landmarks for complete body tracking."""
    important_indices = [0, 11, 12, 13, 14, 23, 24]
    return [(lm.x, lm.y, lm.z) for idx, lm in enumerate(landmarks) if idx in important_indices]

def update_frame(frame_label, pose, frame_callback=None):
    """Capture and process a frame."""
    global cap  # Deklarasi cap sebagai global

    def run_capture():
        global last_frame_time  # Mengakses variabel global last_frame_time
        while True:
            current_time = time()
            if current_time - last_frame_time < frame_interval:
                continue

            last_frame_time = current_time
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            # Resize frame to desired size
            frame = cv2.resize(frame, (320, 240))  # Ukuran frame lebih kecil

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                if frame_callback:
                    frame_callback(results.pose_landmarks.landmark)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            frame_label.imgtk = frame_tk
            frame_label.configure(image=frame_tk)

    thread = threading.Thread(target=run_capture, daemon=True)
    thread.start()


class ScrollableFrame(Frame):
    """A scrollable frame using Canvas."""
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        canvas = Canvas(self, borderwidth=0)
        frame = Frame(canvas)
        self.scrollbar = Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((4, 4), window=frame, anchor="nw", tags="frame")

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.canvas = canvas
        self.frame = frame

class Page(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

class AddDataPage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        self.dance_name_var = StringVar()
        self.motion_score_var = StringVar()
        self.recording_label = None
        self.is_recording = False

        tk.Label(self, text="Tambah Data Gerakan Tari", font=("Helvetica", 16)).pack(pady=10)
        video_label = tk.Label(self)
        video_label.pack()

        form_frame = ScrollableFrame(self)
        form_frame.pack(fill="both", expand=True)

        tk.Button(form_frame.frame, text="Mulai Rekam", command=self.start_recording).pack(pady=5)
        tk.Button(form_frame.frame, text="Berhenti Rekam", command=self.stop_recording).pack(pady=5)

        self.recording_label = tk.Label(form_frame.frame, text="", font=("Helvetica", 12), fg="red")
        self.recording_label.pack(pady=5)

        tk.Label(form_frame.frame, text="Nama Tari:").pack(pady=5)
        tk.Entry(form_frame.frame, textvariable=self.dance_name_var).pack(pady=5)
        tk.Label(form_frame.frame, text="Nilai Gerakan:").pack(pady=5)
        tk.Entry(form_frame.frame, textvariable=self.motion_score_var).pack(pady=5)

        tk.Button(form_frame.frame, text="Simpan Gerakan", command=self.save_motion).pack(pady=5)
        tk.Button(form_frame.frame, text="Kembali", command=lambda: controller.show_frame("HomePage")).pack(pady=5)

        self.pose = mp_pose.Pose()
        update_frame(video_label, self.pose, self.capture_landmarks)

    def start_recording(self):
        global recording, recorded_data
        if self.is_recording:
            messagebox.showinfo("Info", "Perekaman sudah dimulai.")
            return

        recording = True
        recorded_data = []
        self.is_recording = True
        self.recording_label.config(text="Sedang merekam...")
        logger.info("Started recording motion.")

    def stop_recording(self):
        global recording
        if not self.is_recording:
            messagebox.showinfo("Info", "Perekaman belum dimulai.")
            return

        recording = False
        self.is_recording = False
        self.recording_label.config(text="")
        logger.info("Stopped recording motion.")
        messagebox.showinfo("Rekam Selesai", "Perekaman gerakan selesai. Silakan masukkan nama tari dan nilai.")

    def capture_landmarks(self, landmarks):
        if self.is_recording:
            filtered_landmarks = filter_landmarks(landmarks)
            logger.debug(f"Captured landmarks: {filtered_landmarks}")
            recorded_data.append(filtered_landmarks)

    def save_motion(self):
        dance_name = self.dance_name_var.get().strip()
        motion_score = self.motion_score_var.get().strip()

        if not recorded_data:
            messagebox.showwarning("Peringatan", "Tidak ada gerakan yang direkam. Mulai rekam terlebih dahulu.")
            return
        if not dance_name:
            messagebox.showwarning("Peringatan", "Nama tari tidak boleh kosong.")
            return
        if not motion_score.isdigit():
            messagebox.showwarning("Peringatan", "Skor harus berupa angka.")
            return

        add_dance(dance_name)
        dance_id = [dance[0] for dance in get_all_dances() if dance[1] == dance_name][0]
        add_motion(dance_id, recorded_data, float(motion_score))
        evaluate_page = self.controller.frames["EvaluatePage"]
        evaluate_page.update_dropdown()
        messagebox.showinfo("Sukses", f"Gerakan disimpan untuk tari '{dance_name}'.")
        
class EvaluatePage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        self.selected_dance_var = StringVar()
        self.is_evaluating = False
        self.best_score = 0

        tk.Label(self, text="Penilaian Gerakan Tari", font=("Helvetica", 16)).pack(pady=10)
        video_label = tk.Label(self)
        video_label.pack()

        dropdown_frame = ScrollableFrame(self)
        dropdown_frame.pack(fill="both", expand=True)

        tk.Label(dropdown_frame.frame, text="Pilih Tari:").pack(pady=5)
        self.dropdown = ttk.Combobox(dropdown_frame.frame, textvariable=self.selected_dance_var, state="readonly")
        self.dropdown.pack(pady=5)
        self.update_dropdown()

        tk.Button(dropdown_frame.frame, text="Mulai Evaluasi", command=self.start_evaluation).pack(pady=5)
        tk.Button(dropdown_frame.frame, text="Berhenti Evaluasi", command=self.stop_evaluation).pack(pady=5)

        self.evaluating_label = tk.Label(dropdown_frame.frame, text="", font=("Helvetica", 12), fg="red")
        self.evaluating_label.pack(pady=5)

        tk.Button(dropdown_frame.frame, text="Kembali", command=lambda: controller.show_frame("HomePage")).pack(pady=5)

        self.pose = mp_pose.Pose()
        update_frame(video_label, self.pose, self.evaluate_frame_callback)

    def start_evaluation(self):
        dance_name = self.selected_dance_var.get()
        if not dance_name:
            messagebox.showwarning("Peringatan", "Pilih tari terlebih dahulu.")
            return

        self.dance_id = [dance[0] for dance in get_all_dances() if dance[1] == dance_name][0]
        self.is_evaluating = True
        self.best_score = 0
        recorded_data.clear()
        self.evaluating_label.config(text="Sedang mengevaluasi...")
        logger.info("Started evaluation.")

    def stop_evaluation(self):
        if not self.is_evaluating:
            messagebox.showinfo("Info", "Evaluasi belum dimulai.")
            return

        self.is_evaluating = False
        self.evaluating_label.config(text="")
        logger.info("Stopped evaluation.")
        messagebox.showinfo("Hasil Evaluasi", f"Evaluasi selesai. Skor terbaik: {self.best_score}")

    def evaluate_frame_callback(self, landmarks):
        if self.is_evaluating:
            filtered_landmarks = filter_landmarks(landmarks)
            recorded_data.append(filtered_landmarks)

            # Calculate score dynamically
            motions = get_dance_motions(self.dance_id)
            for motion in motions:
                reference_landmarks = json.loads(motion[0])
                score = calculate_score(reference_landmarks, recorded_data)
                self.best_score = max(self.best_score, score)
            logger.debug(f"Current best score: {self.best_score}")

    def update_dropdown(self):
        dances = get_all_dances()
        self.dropdown["values"] = [dance[1] for dance in dances]


class HomePage(Page):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        tk.Label(self, text="Sistem Penilaian Gerakan Tari", font=("Helvetica", 20)).pack(pady=20)
        tk.Button(self, text="Tambah Data Tari", command=lambda: controller.show_frame("AddDataPage")).pack(pady=10)
        tk.Button(self, text="Penilaian Tari", command=lambda: controller.show_frame("EvaluatePage")).pack(pady=10)
        tk.Button(self, text="Keluar", command=self.quit_application).pack(pady=10)

    def quit_application(self):
        release_resources()
        self.controller.destroy()

class DanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dance Evaluation App")
        self.geometry("800x600")

        self.frames = {}
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        for PageClass in (HomePage, AddDataPage, EvaluatePage):
            page_name = PageClass.__name__
            frame = PageClass(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

def release_resources():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Resources released.")

if __name__ == "__main__":
    logger.info("Starting application.")
    initialize_database()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Gagal membuka kamera. Pastikan kamera terhubung.")
        exit()

    app = DanceApp()
    atexit.register(release_resources)
    app.mainloop()
