import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageTk
import os
from twilio.rest import Client
import time

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analysis App")
        self.root.geometry("1000x700") 
        self.root.minsize(1000, 700)
        self.root.maxsize(1000, 700)
        self.root.configure(bg="#21F5E5") 

        self.bg_image = Image.open("/Users/harshi/Desktop/opencv_project/bg1.jpeg")
        self.bg_image = self.bg_image.resize((1000, 700))  # Resize image as needed
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Create a label to display the background image
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Create a Label with large text
        large_text = tk.Label(root, text="WATCHFULEYE : ABNORMAL HUMAN ACTIVITY DETECTION", font=("Georgia", 27,"bold"),bg="#ADD8E6",fg="black")
        large_text.pack(pady=20)

        self.model_path = '/Users/harshi/Downloads/AHAR_lesshighvideos_savedmodel_epoch20_accuracy71hack.h5'
        # self.model_path = '/Users/harshi/Downloads/AHAR_lesshighvideos_savedmodel_epoch10_accuracy.h5'
        self.model = load_model(self.model_path)

        self.class_mapping = {
            'HighDanger': 3,
            'LowDanger': 1,
            'MediumDanger': 2,
            'Normal': 0
        }


        self.cap = None
        self.video_label = tk.Label(root, text="No video selected",bg="red", fg="white",font=("Georgia", 20,"bold"))
        self.video_label.pack(pady=30)

        self.select_button = tk.Button(root, text="Select Video", command=self.select_video,font=("Georgia", 20,"bold"),highlightbackground="#ADD8E6")
        self.select_button.pack(pady=20)

        self.start_button = tk.Button(root, text="Start Analysis", command=self.start_analysis,font=("Georgia", 20,"bold"),highlightbackground="#ADD8E6")
        self.start_button.pack(pady=20)

        self.stop_button = tk.Button(root, text="Quit App", command=self.stop_analysis,font=("Georgia", 20,"bold"),highlightbackground="#ADD8E6")
        self.stop_button.pack(pady=20)


        # Twilio credentials and phone numbers
        self.twilio_account_sid = Acc_id
        self.twilio_auth_token = Auth_token
        self.from_number = From_number
        self.to_number = To_number

    def select_video(self):
        file_path = filedialog.askopenfilename(parent=self.root, filetypes=[("Video files", "*.mp4")])
        max_char=10
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.video_label.config(text=f"Selected Video: {file_path[:max_char]}",bg="green", fg="white",font=("Georgia", 20,"bold"))
            # Adjust the size of the main window
            self.root.geometry("1000x700")

    def start_analysis(self):
        if self.cap is not None:
            self.play_original_video()

    def stop_analysis(self):
        self.root.quit()  # Stop the Tkinter event loop


    def send_alert_message(self):
        client = Client(self.twilio_account_sid, self.twilio_auth_token)
        message = client.messages.create(
            body="Danger level high for 4 seconds!",
            from_=self.from_number,
            to=self.to_number
        )
        print("Alert message sent:", message.sid)

    def play_original_video(self):
        consecutive_high_danger_frames = 0
        start_time = time.time()

        cv2.namedWindow('Analysing Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Analysing Video', 600, 600) 
        cv2.moveWindow('Analysing Video', 100, 100)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            # Resize the frame to match the expected input shape of the model
            resized_frame = cv2.resize(frame, (160, 160))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            preprocessed_frame = img_to_array(gray_frame)
            preprocessed_frame = preprocess_input(preprocessed_frame)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

            predictions = self.model.predict(preprocessed_frame)
            predicted_label_index = np.argmax(predictions)
            keys_list = list(self.class_mapping.keys())
            key_at_index = keys_list[predicted_label_index]
            value_at_index = self.class_mapping[key_at_index]

            # Overlay prediction on the frame
            cv2.putText(display_frame, f'Danger Level: {key_at_index}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Check if danger level is high (2 or 3)
            if value_at_index in [2, 3]:
                consecutive_high_danger_frames += 1
                if consecutive_high_danger_frames >= 120:  # 120 frames at 30fps = 4 seconds
                    current_time = time.time()
                    if current_time - start_time >= 4:
                        # Send alert message
                        self.send_alert_message()
                        # Reset variables
                        consecutive_high_danger_frames = 0
                        start_time = current_time
            else:
                consecutive_high_danger_frames = 0
                start_time = time.time()

            cv2.imshow('Analysing Video', display_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()

