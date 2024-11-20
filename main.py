import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_posture(landmarks):
    """
    Mendeteksi postur tubuh berdasarkan landmark dari MediaPipe Pose.
    """
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y

    if abs(shoulder_y - hip_y) < 0.1 and abs(hip_y - knee_y) < 0.1:
        return "Tidur"
    elif abs(shoulder_y - hip_y) > 0.2 and abs(hip_y - knee_y) > 0.2:
        return "Berdiri"
    else:
        return "Duduk"

def update_frame():
    """
    Fungsi untuk menangkap frame dari kamera, mendeteksi postur, dan memperbarui UI.
    """
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses kamera.")
        return

    # Konversi frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    posture = "Tidak Terdeteksi"
    if results.pose_landmarks:
        # Dapatkan landmark dan deteksi postur
        landmarks = results.pose_landmarks.landmark
        posture = detect_posture(landmarks)

        # Gambar stikman di frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    # Periksa apakah postur sesuai dengan pilihan
    if selected_posture.get() != "All" and posture != selected_posture.get():
        posture = "Tidak Sesuai"

    # Tambahkan indikator deteksi
    color_map = {
        "Berdiri": (0, 255, 0),  # Hijau
        "Duduk": (255, 255, 0),  # Kuning
        "Tidur": (255, 0, 0),    # Merah
        "Tidak Terdeteksi": (0, 0, 255),  # Merah (default)
        "Tidak Sesuai": (255, 0, 255),    # Magenta
    }
    color = color_map.get(posture, (255, 255, 255))  # Default putih

    cv2.putText(frame, f"Postur: {posture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

    # Konversi frame untuk tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    # Tampilkan frame di label
    video_label.imgtk = frame_tk
    video_label.configure(image=frame_tk)
    video_label.after(10, update_frame)

def exit_app():
    """Fungsi untuk keluar dari aplikasi."""
    cap.release()
    root.destroy()

# Inisialisasi aplikasi tkinter
root = tk.Tk()
root.title("Deteksi Postur Tubuh")
root.geometry("800x600")

# Variabel global untuk memilih mode
selected_posture = StringVar(value="All")

# Label video
video_label = tk.Label(root)
video_label.pack()

# Pilihan mode deteksi
frame_controls = tk.Frame(root)
frame_controls.pack(pady=10)

tk.Label(frame_controls, text="Pilih Postur: ").grid(row=0, column=0, padx=5)
modes = ["All", "Berdiri", "Duduk", "Tidur"]
for i, mode in enumerate(modes):
    tk.Radiobutton(frame_controls, text=mode, variable=selected_posture, value=mode).grid(row=0, column=i+1, padx=5)

# Tombol keluar
exit_button = tk.Button(root, text="Keluar", command=exit_app, bg="red", fg="white")
exit_button.pack(pady=20)

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    root.destroy()

# Inisialisasi MediaPipe Pose
pose = mp_pose.Pose()

# Jalankan pembaruan frame
update_frame()

# Jalankan aplikasi tkinter
root.mainloop()
