# Sign Language Recognition System 🤟

A comprehensive web-based application designed to bridge the communication gap for the hearing and speech impaired. This system uses computer vision and deep learning to translate American Sign Language (ASL) alphabets and predefined words into text and speech in real-time.

![System Screenshot](screenshots/main_interface.png) *(Note: Add your screenshots to the screenshots directory)*

## ✨ Key Features

- **Real-time Recognition**: Instant translation of hand signs using MediaPipe and PyTorch.
- **Dual Mode Support**:
  - **Alphabet Mode**: Recognizes individual ASL characters.
  - **Word Mode**: Recognizes complete words for faster communication.
- **Sentence Construction**: Built-in stability threshold to prevent flickering and accurately build sentences.
- **Text-to-Speech (TTS)**: Integrated speech synthesis to vocalize translated sentences.
- **User Authentication**: Secure login and registration system with **Email OTP Verification**.
- **History Management**: Registered users can save, view, and manage their translation history.
- **Responsive Dashboard**: A modern, dark-themed user interface built with Flask and Vanilla CSS.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: PyTorch (Word Recognition), Scikit-learn (Alphabet Recognition), MediaPipe (Hand Tracking)
- **Computer Vision**: OpenCV
- **Database**: MongoDB
- **Security**: Flask-Bcrypt, SMTP (Email OTP)
- **Speech**: gTTS (Google Text-to-Speech)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- MongoDB (Local or Atlas)
- Webcam for real-time detection

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

3. **Environment Configuration:**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   SECRET_KEY=your_secret_key_here
   MONGO_URI=mongodb://localhost:27017/sign_lang_db
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASS=your-app-specific-password
   ```
   > [!TIP]
   > For Gmail, you'll need to generate an **App Password** from your Google Account settings.

4. **Run the application:**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your web browser.

## 📁 Project Structure

```text
├── app.py              # Main Flask application
├── detect_sign.py      # Standalone detection script
├── action_model.pth    # Trained PyTorch model (Word Recognition)
├── model/              # Alphabet recognition models (Joblib)
├── static/             # CSS, JS, and Generated Audio
├── templates/          # HTML Templates
├── requirement.txt     # Python Dependencies
└── .env                # Environment Variables
```

## 🧠 How it Works

1. **Detection**: MediaPipe extracts 21 hand landmarks (X, Y, Z coordinates).
2. **Pre-processing**: Coordinates are normalized (centered and scaled) to ensure consistency regardless of hand distance from the camera.
3. **Inference**:
   - **Word Mode**: A PyTorch Neural Network processes 126 features (63 per hand).
   - **Alphabet Mode**: A Scikit-learn Random Forest/SVM model processes 63 features.
4. **Action**: The predicted label is added to the current sentence once a stability threshold is met.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Google Text-to-Speech](https://pypi.org/project/gTTS/) for vocalization.
