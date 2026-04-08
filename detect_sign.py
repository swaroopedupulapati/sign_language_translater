import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            pts = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[idx].landmark])
            
            # 1. CENTER: Subtract wrist (point 0)
            pts = pts - pts[0]
            
            # 2. SCALE: Divide by hand scale (distance from wrist to middle finger base)
            scale = np.linalg.norm(pts[9]) # Point 9 is middle finger MCP
            if scale > 0:
                pts = pts / scale
                
            landmarks = pts.flatten()
            if label == 'Left':
                lh = landmarks
            else:
                rh = landmarks
    return np.concatenate([lh, rh])

# Define Model (must match train_model.py)
class StaticSignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StaticSignModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def main():
    # 1. Load the model
    model_path = os.path.join(os.path.dirname(__file__), 'action_model.pth')
    try:
        # Using weights_only=False because the model contains numpy objects (actions list)
        checkpoint = torch.load(model_path, weights_only=False)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    actions = checkpoint['actions']
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']

    model = StaticSignModel(input_size, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(0)
    threshold = 0.7

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, hands)
            draw_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            
            with torch.no_grad():
                input_vec = torch.FloatTensor(np.expand_dims(keypoints, axis=0))
                res = model(input_vec)
                res_probs = torch.softmax(res, dim=1).squeeze().numpy()
                pred_idx = np.argmax(res_probs)
                
                prediction = "..."
                if res_probs[pred_idx] > threshold:
                    prediction = actions[pred_idx]

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, prediction, (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Static Sign Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
