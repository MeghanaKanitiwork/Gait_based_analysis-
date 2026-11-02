import cv2
import numpy as np

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Open the video file
video_path = 'B:/gait_based_analysis/Video 7.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the mask to get binary silhouettes
    _, silhouettes = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Display the silhouettes
    cv2.imshow('Silhouettes', silhouettes)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import os
# import numpy as np
# import mediapipe as mp
# from sklearn.svm import OneClassSVM
# from sklearn.model_selection import train_test_split

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load and preprocess the training data
# def load_silhouettes(input_dir, target_size=(128, 256)):
#     silhouettes = []
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.png'):
#             img_path = os.path.join(input_dir, filename)
#             print(f'Reading file: {img_path}')  # Debugging line
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f'Error reading file: {img_path}')  # Debugging line
#                 continue
#             resized_img = cv2.resize(img, target_size)
#             silhouettes.append(resized_img.flatten())
#     return np.array(silhouettes)

# # Example usage
# input_dir = 'B:/gait_based_analysis/Normalized_silhouettes'
# silhouettes = load_silhouettes(input_dir)
# print(f'Loaded {len(silhouettes)} silhouettes.')

# # Split the data into training and test sets
# X_train, X_test = train_test_split(silhouettes, test_size=0.2, random_state=42)

# # Train the One-Class SVM model
# model = OneClassSVM(gamma='auto').fit(X_train)

# def process_gesture(landmarks):
#     # Function to classify gesture based on hand landmarks
#     return "Detected Gesture"

# def draw_gait_lines(frame, landmarks):
#     # Extract key points
#     head = landmarks[mp_pose.PoseLandmark.NOSE]
#     shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#     shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#     elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
#     elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
#     knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
#     knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

#     # Draw lines connecting key points
#     def draw_line(p1, p2, color=(0, 255, 0), thickness=2):
#         cv2.line(frame, (int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])),
#                  (int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])), color, thickness)

#     draw_line(head, shoulder_left)
#     draw_line(head, shoulder_right)
#     draw_line(shoulder_left, elbow_left)
#     draw_line(shoulder_right, elbow_right)
#     draw_line(elbow_left, knee_left)
#     draw_line(elbow_right, knee_right)

# # Initialize video capture
# cap = cv2.VideoCapture('B:/gait_based_analysis/Video 1.mp4')

# # Background subtractor for silhouette extraction
# bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the image to RGB
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results_pose = pose.process(image_rgb)
#     results_hands = hands.process(image_rgb)

#     # Face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     # Draw pose landmarks and connections
#     if results_pose.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         draw_gait_lines(frame, results_pose.pose_landmarks.landmark)

#     # Draw hand landmarks and connections
#     if results_hands.multi_hand_landmarks:
#         for landmarks in results_hands.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
#             gesture = process_gesture(landmarks)
#             cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # Apply background subtraction for silhouette extraction
#     fg_mask = bg_subtractor.apply(frame)
#     _, silhouette = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

#     # Resize and flatten the silhouette for anomaly detection
#     resized_silhouette = cv2.resize(silhouette, (128, 256)).flatten().reshape(1, -1)
#     anomaly_prediction = model.predict(resized_silhouette)
#     anomaly_label = 'normal' if anomaly_prediction == 1 else 'anomaly'

#     # Display the silhouette and anomaly label
#     cv2.putText(frame, f'Anomaly: {anomaly_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Silhouette', silhouette)
#     cv2.imshow('Threat Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load and preprocess the training data
def load_silhouettes(input_dir, target_size=(128, 256)):
    silhouettes = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            print(f'Reading file: {img_path}')  # Debugging line
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Error reading file: {img_path}')  # Debugging line
                continue
            resized_img = cv2.resize(img, target_size)
            silhouettes.append(resized_img.flatten())
    return np.array(silhouettes)

# Example usageq
input_dir = 'B:/gait_based_analysis/Normalized_silhouettes'
silhouettes = load_silhouettes(input_dir)
print(f'Loaded {len(silhouettes)} silhouettes.')

# Split the data into training and test sets
X_train, X_test = train_test_split(silhouettes, test_size=0.2, random_state=42)

# Train the One-Class SVM model
model = OneClassSVM(gamma='auto').fit(X_train)

def process_gesture(landmarks):
    # Function to classify gesture based on hand landmarks
    return "Detected Gesture"

def draw_gait_lines(frame, landmarks):
    # Extract key points
    head = landmarks[mp_pose.PoseLandmark.NOSE]
    shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    # Draw lines connecting key points
    def draw_line(p1, p2, color=(0, 255, 0), thickness=2):
        cv2.line(frame, (int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])),
                 (int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])), color, thickness)

    draw_line(head, shoulder_left)
    draw_line(head, shoulder_right)
    draw_line(shoulder_left, elbow_left)
    draw_line(shoulder_right, elbow_right)
    draw_line(elbow_left, knee_left)
    draw_line(elbow_right, knee_right)

# Initialize video capture
cap = cv2.VideoCapture('B:/gait_based_analysis/Video 7.mp4')

# Background subtractor for silhouette extraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)

    # Draw pose landmarks and connections
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_gait_lines(frame, results_pose.pose_landmarks.landmark)

    # Draw hand landmarks and connections
    if results_hands.multi_hand_landmarks:
        for landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = process_gesture(landmarks)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Apply background subtraction for silhouette extraction
    fg_mask = bg_subtractor.apply(frame)
    _, silhouette = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Resize and flatten the silhouette for anomaly detection
    resized_silhouette = cv2.resize(silhouette, (128, 256)).flatten().reshape(1, -1)
    anomaly_prediction = model.predict(resized_silhouette)
    anomaly_label = 'normal' if anomaly_prediction == 1 else 'anomaly'

    # Display the silhouette and anomaly label
    cv2.putText(frame, f'Anomaly: {anomaly_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Silhouette', silhouette)
    cv2.imshow('Threat Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
