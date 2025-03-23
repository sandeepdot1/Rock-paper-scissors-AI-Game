import cv2
import numpy as np
import tensorflow.lite as tflite
import threading
import time
from random import choice

# Reverse Mapping
REV_CLASS_MAP = {
    0: "none",
    1: "paper",
    2: "rock",
    3: "scissors"
}

def mapper(val):
    return REV_CLASS_MAP[val]

# Winner Calculation
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    outcomes = {("rock", "scissors"): "User", ("scissors", "paper"): "User", ("paper", "rock"): "User"}
    return outcomes.get((move1, move2), "Computer")

# Load TensorFlow Lite Model
interpreter = tflite.Interpreter(model_path="model_stone_paper_scissors.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video Capture in Separate Thread
class VideoStream:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.ret, self.frame = self.video.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.video.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.video.release()

# Start Video Stream
video_stream = VideoStream()
prev_move = None
computer_move_name = "none"
winner = "Waiting..."

# Preload Computer Move Icons
move_icons = {
    "rock": cv2.imread("images/rock.png"),
    "paper": cv2.imread("images/paper.png"),
    "scissors": cv2.imread("images/scissors.png")
}

# Function to Predict Move using TFLite
def predict_move(img):
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    return mapper(np.argmax(pred[0]))

# Main Loop
while True:
    start_time = time.time()

    frame = video_stream.get_frame()
    frame = cv2.resize(frame, (1300, 720))

    # User Region of Interest
    roi = frame[100:600, 100:600]
    user_move_name = predict_move(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Determine Computer Move and Winner
    if prev_move != user_move_name and user_move_name != "none":
        computer_move_name = choice(["rock", "paper", "scissors"])
        winner = calculate_winner(user_move_name, computer_move_name)

    prev_move = user_move_name

    # Draw Border around User's Gesture Area
    cv2.rectangle(frame, (100, 100), (600, 600), (0, 255, 0), 3)  # Green Border for Hand Placement

    # Display Information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Your Move: {user_move_name}", (50, 50), font, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer's Move: {computer_move_name}", (750, 50), font, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Winner: {winner}", (400, 700), font, 2, (0, 0, 255), 4)

    # Display Computer's Move Icon
    if computer_move_name in move_icons and move_icons[computer_move_name] is not None:
        icon = cv2.resize(move_icons[computer_move_name], (500, 500))
        frame[100:600, 750:1250] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    # Break Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Frame Time: {time.time() - start_time:.3f} sec")  # Check FPS

# Clean Up
video_stream.stop()
cv2.destroyAllWindows()
