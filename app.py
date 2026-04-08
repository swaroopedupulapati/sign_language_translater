import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import threading
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob
import flask
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import random
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")
bcrypt = Bcrypt(app)

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client['sign_lang_db']
users_collection = db['users']
history_collection = db['history']

# Email Configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def send_otp_email(receiver_email, otp):
    try:
        subject = "Your OTP for Sign Language App"
        body = f"Your OTP for registration/password reset is: {otp}. It will expire in 10 minutes."
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = receiver_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Global variables
predicted_sentence = ""
current_sign = ""  # Word
current_alphabet = ""  # Letter
prediction_count = 0
threshold_frames = 15
last_prediction = ""
camera_active = False
current_mode = "word"  # Default mode: "word" or "alphabet"

# Create static directory for audio files if it doesn't exist
AUDIO_DIR = os.path.join('static', 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Define Model (must match detect_sign.py)
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

# Load the model and actions
model_path = os.path.join(os.path.dirname(__file__), 'action_model.pth')
model = None
actions = []
model_loaded = False

try:
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        actions = checkpoint['actions']
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        
        model = StaticSignModel(input_size, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_loaded = True
        print("PyTorch model loaded successfully.")
    else:
        print(f"Error: Model not found at {model_path}")
except Exception as e:
    print(f"Warning: failed to load model: {e}")
    model_loaded = False

# Load Alphabet model (joblib)
alp_model_path = os.path.join('model', 'asl_model.joblib')
alp_encoder_path = os.path.join('model', 'label_encoder.joblib')
alp_model = None
le = None
alp_model_loaded = False

try:
    if os.path.exists(alp_model_path) and os.path.exists(alp_encoder_path):
        alp_model = joblib.load(alp_model_path)
        le = joblib.load(alp_encoder_path)
        alp_model_loaded = True
        print("Alphabet model (joblib) loaded successfully.")
    else:
        print(f"Warning: Alphabet model/encoder not found at {alp_model_path}")
except Exception as e:
    print(f"Warning: failed to load alphabet model: {e}")
    alp_model_loaded = False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(result):
    """Exact logic from detect_sign.py for Word Recognition"""
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if result.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(result.multi_handedness):
            label = hand_handedness.classification[0].label
            pts = np.array([[res.x, res.y, res.z] for res in result.multi_hand_landmarks[idx].landmark])
            
            # 1. CENTER: Subtract wrist (point 0)
            pts = pts - pts[0]
            
            # 2. SCALE: Subtract palm distance (using point 9 middle MCP)
            scale = np.linalg.norm(pts[9])
            if scale > 0:
                pts = pts / scale
                
            landmarks = pts.flatten()
            if label == 'Left':
                lh = landmarks
            else:
                rh = landmarks
    return np.concatenate([lh, rh])

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global predicted_sentence, current_sign, current_alphabet, prediction_count, last_prediction
        
        # Capture frame (NO FLIP before detection to match detect_sign.py)
        ret, frame = self.video.read()
        if not ret:
            return None
            
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            # ONLY run the inference for the active mode
            if current_mode == "alphabet" and alp_model_loaded:
                # Alphabet model (Joblib) expects raw landmarks from FIRST hand only (63 features)
                first_hand = result.multi_hand_landmarks[0]
                features = []
                for lm in first_hand.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                x_input = np.array(features).reshape(1, -1)
                alp_pred = alp_model.predict(x_input)
                current_alphabet = le.inverse_transform(alp_pred)[0]
                current_sign = "nothing" # Word model not active
                label_to_track = current_alphabet
            
            elif current_mode == "word" and model_loaded:
                # Word model (PyTorch) expects 126 features centered/scaled from both hands
                keypoints = extract_keypoints(result)
                with torch.no_grad():
                    input_vec = torch.FloatTensor(np.expand_dims(keypoints, axis=0))
                    res = model(input_vec)
                    res_probs = torch.softmax(res, dim=1).squeeze().numpy()
                    pred_idx = np.argmax(res_probs)
                    
                    threshold = 0.7
                    if res_probs[pred_idx] > threshold:
                        current_word_label = actions[pred_idx]
                    else:
                        current_word_label = "nothing"
                
                current_sign = current_word_label
                current_alphabet = "nothing" # Alphabet model not active
                label_to_track = current_word_label
            else:
                label_to_track = "nothing"

            # Auto-stability for sentence building
            if label_to_track == last_prediction:
                prediction_count += 1
            else:
                prediction_count = 0
                last_prediction = label_to_track
            
            if prediction_count == threshold_frames:
                if label_to_track == "space":
                    predicted_sentence += " "
                elif label_to_track == "del":
                    predicted_sentence = predicted_sentence[:-1]
                elif label_to_track != "nothing":
                    predicted_sentence += label_to_track
                    # Add space ONLY for words
                    if current_mode == "word":
                        predicted_sentence += " "
                prediction_count = 0

            # Draw landmarks
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            current_sign = "nothing"
            current_alphabet = "nothing"
            prediction_count = 0
            last_prediction = ""
        
        # Flip the frame for natural UI preview ONLY AFTER detection/inference
        frame = cv2.flip(frame, 1)

        # Add text overlay
        cv2.rectangle(frame, (10, 10), (630, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Word: {current_sign}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Alp: {current_alphabet}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Sent: {predicted_sentence}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        return frame

def gen_frames():
    camera = None
    try:
        camera = VideoCamera()
        while camera_active:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        try:
            if camera is not None and hasattr(camera, 'video'):
                camera.video.release()
        except Exception:
            pass

def cleanup_old_audio_files():
    """Remove audio files older than 1 hour"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=1)
        audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        
        for file_path in audio_files:
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old audio file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

def generate_audio_file(text):
    """Generate MP3 file from text using gTTS"""
    try:
        if not text or text.strip() == "":
            return None
            
        # Clean up old files first
        cleanup_old_audio_files()
        
        # Generate unique filename
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.route('/')
@login_required
def index():
    return render_template('index.html', 
                          user_email=session.get('user_email'),
                          user_name=session.get('user_name'))

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users_collection.find_one({'email': email, 'verified': True}):
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        otp = str(random.randint(100000, 999999))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Update or insert unverified user
        users_collection.update_one(
            {'email': email},
            {'$set': {
                'name': name,
                'password': hashed_password,
                'otp': otp,
                'otp_expiry': datetime.now() + timedelta(minutes=10),
                'verified': False
            }},
            upsert=True
        )
        
        if send_otp_email(email, otp):
            session['pending_email'] = email
            return redirect(url_for('verify_otp'))
        else:
            flash('Failed to send OTP. Please try again.', 'danger')
            
    return render_template('register.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get('pending_email')
    if not email:
        return redirect(url_for('register'))
        
    if request.method == 'POST':
        otp_input = request.form.get('otp')
        user = users_collection.find_one({'email': email})
        
        if user and user.get('otp') == otp_input and datetime.now() < user.get('otp_expiry'):
            users_collection.update_one({'email': email}, {'$set': {'verified': True}, '$unset': {'otp': 1, 'otp_expiry': 1}})
            session.pop('pending_email', None)
            session['user_email'] = email
            session['user_name'] = user.get('name')
            flash('Registration successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid or expired OTP', 'danger')
            
    return render_template('verify_otp.html', email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users_collection.find_one({'email': email, 'verified': True})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_email'] = email
            session['user_name'] = user.get('name')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = users_collection.find_one({'email': email, 'verified': True})
        
        if user:
            otp = str(random.randint(100000, 999999))
            users_collection.update_one(
                {'email': email},
                {'$set': {
                    'otp': otp,
                    'otp_expiry': datetime.now() + timedelta(minutes=10)
                }}
            )
            if send_otp_email(email, otp):
                session['reset_email'] = email
                return redirect(url_for('reset_password'))
            else:
                flash('Failed to send OTP', 'danger')
        else:
            flash('Email not found', 'danger')
            
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    email = session.get('reset_email')
    if not email:
        return redirect(url_for('forgot_password'))
        
    if request.method == 'POST':
        otp_input = request.form.get('otp')
        new_password = request.form.get('password')
        user = users_collection.find_one({'email': email})
        
        if user and user.get('otp') == otp_input and datetime.now() < user.get('otp_expiry'):
            hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            users_collection.update_one(
                {'email': email},
                {'$set': {'password': hashed_password}, '$unset': {'otp': 1, 'otp_expiry': 1}}
            )
            session.pop('reset_email', None)
            flash('Password reset successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired OTP', 'danger')
            
    return render_template('reset_password.html', email=email)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode, prediction_count, last_prediction
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    if mode in ["word", "alphabet"]:
        current_mode = mode
        prediction_count = 0  # Reset stability when switching
        last_prediction = ""
        return jsonify({'status': 'success', 'mode': current_mode})
    return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    return jsonify({
        'sentence': predicted_sentence,
        'current_sign': current_sign,
        'current_alphabet': current_alphabet,
        'prediction_count': prediction_count,
        'threshold_frames': threshold_frames
    })

@app.route('/recognize_word', methods=['POST'])
def recognize_word():
    """Manual trigger to recognize the current SIGN/WORD and add it to the sentence"""
    global predicted_sentence, current_sign
    if current_sign and current_sign != "nothing":
        if current_sign == "space":
            predicted_sentence += " "
        elif current_sign == "del":
            predicted_sentence = predicted_sentence[:-1]
        else:
            predicted_sentence += current_sign + " "  # Append space after word
        return jsonify({'status': 'success', 'word': current_sign, 'sentence': predicted_sentence})
    return jsonify({'status': 'error', 'message': 'No word detected'})

@app.route('/recognize_alphabet', methods=['POST'])
def recognize_alphabet():
    """Manual trigger to recognize the current ALPHABET and add it to the sentence"""
    global predicted_sentence, current_alphabet
    if current_alphabet and current_alphabet != "nothing":
        if current_alphabet == "space":
            predicted_sentence += " "
        elif current_alphabet == "del":
            predicted_sentence = predicted_sentence[:-1]
        else:
            predicted_sentence += current_alphabet
        return jsonify({'status': 'success', 'alphabet': current_alphabet, 'sentence': predicted_sentence})
    return jsonify({'status': 'error', 'message': 'No alphabet detected'})


@app.route('/set_sentence', methods=['POST'])
def set_sentence():
    """Set the current sentence from website input (JSON: {'text': '...'})"""
    global predicted_sentence
    data = request.get_json(silent=True) or {}
    text = data.get('text') if isinstance(data, dict) else None
    if not text or not isinstance(text, str):
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    predicted_sentence = text.strip()
    return jsonify({'status': 'success', 'sentence': predicted_sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    return jsonify({'status': 'Sentence cleared'})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global predicted_sentence
    
    # Check if custom text is provided in request body
    data = request.get_json(silent=True) or {}
    text_to_speak = data.get('text') or predicted_sentence
    
    if not text_to_speak or text_to_speak.strip() == "":
        return jsonify({
            'status': 'error',
            'message': 'No sentence to speak'
        }), 400
    
    # Generate audio file
    audio_filename = generate_audio_file(text_to_speak)
    
    if audio_filename:
        return jsonify({
            'status': 'success',
            'sentence': text_to_speak,
            'audio_url': f'/static/audio/{audio_filename}',
            'message': 'Audio generated successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate audio'
        }), 500

@app.route('/save_history', methods=['POST'])
@login_required
def save_history():
    global predicted_sentence
    if not predicted_sentence or predicted_sentence.strip() == "":
        return jsonify({'status': 'error', 'message': 'No sentence to save'}), 400
        
    try:
        history_item = {
            'user_email': session.get('user_email'),
            'sentence': predicted_sentence.strip(),
            'timestamp': datetime.now()
        }
        history_collection.insert_one(history_item)
        return jsonify({'status': 'success', 'message': 'Sentence saved to history'})
    except Exception as e:
        print(f"Error saving history: {e}")
        return jsonify({'status': 'error', 'message': 'Database error'}), 500

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    try:
        user_email = session.get('user_email')
        history_items = list(history_collection.find({'user_email': user_email}).sort('timestamp', -1))
        
        # Convert ObjectId to string for JSON serialization and format timestamp
        for item in history_items:
            item['_id'] = str(item['_id'])
            item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
        return jsonify({'status': 'success', 'history': history_items})
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'status': 'error', 'message': 'Database error'}), 500

@app.route('/delete_history_item', methods=['POST'])
@login_required
def delete_history_item():
    try:
        data = request.get_json(silent=True) or {}
        item_id = data.get('id')
        if not item_id:
            return jsonify({'status': 'error', 'message': 'No ID provided'}), 400
            
        result = history_collection.delete_one({
            '_id': ObjectId(item_id),
            'user_email': session.get('user_email')
        })
        
        if result.deleted_count > 0:
            return jsonify({'status': 'success', 'message': 'Item deleted'})
        else:
            return jsonify({'status': 'error', 'message': 'Item not found or unauthorized'}), 404
    except Exception as e:
        print(f"Error deleting history: {e}")
        return jsonify({'status': 'error', 'message': 'Database error'}), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
