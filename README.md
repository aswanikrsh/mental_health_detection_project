# 🧠 Mental Health Risk Detection and Alert System

This project is a web-based application that detects potential suicidal intent in user-submitted text using a machine learning model. If high-risk content is detected, the system immediately sends an SMS and initiates a call to a registered emergency contact using Twilio.

## 🚀 Features

- Suicide vs. Non-suicide text classification
- Web interface built with Flask
- TF-IDF vectorization + Naive Bayes classifier
- Handles class imbalance with SMOTE
- Real-time SMS and voice call alerts via Twilio
- “Get Help” page with resources

## 🧰 Tech Stack

- **Python**
- **Scikit-learn**
- **Flask**
- **Twilio**
- **Pandas**
- **SMOTE (from imbalanced-learn)**

## 📁 Project Structure

```
├── app.py                   # Flask app
├── train.py                 # Model training script
├── Suicide_Detection.csv    # Dataset
├── mental_health_model.pkl  # Trained model
├── tfidf_vectorizer.pkl     # Saved vectorizer
├── templates/
│   ├── index.html           # Main UI
│   └── help.html            # Help/resources page
```

## 📊 Dataset

- **Source**: [Kaggle Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Total entries**: ~23,000 messages
- **Classes**: `suicide`, `non-suicide`
- **Features**: Raw text data
- **Target**: Suicide risk label

## ⚙️ How It Works

1. **Training** (`train.py`)
   - Clean text data
   - Convert text to TF-IDF vectors
   - Use SMOTE to handle class imbalance
   - Train a Naive Bayes model
   - Save model and vectorizer using `pickle`

2. **Prediction + Alerting** (`app.py`)
   - User inputs text
   - Text is vectorized and predicted
   - If risk is high:
     - SMS is sent
     - Voice call is triggered

## 🛠 Setup Instructions

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/mental-health-alert-system.git
   cd mental-health-alert-system
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**  
   ```bash
   python train.py
   ```

4. **Run the app**  
   ```bash
   python app.py
   ```

5. **Go to** `http://127.0.0.1:5000`

## 🔐 Twilio Setup

- Sign up at [https://www.twilio.com](https://www.twilio.com)
- Get your **Account SID**, **Auth Token**, and **Twilio phone number**
- Replace them in `app.py`:

```python
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = '+1234567890'
EMERGENCY_PHONE_NUMBER = '+91xxxxxxxxxx'
```

## 📈 Model Performance

- **Accuracy**: ~90%  
- **Precision/Recall/F1**: Reported in `train.py` during evaluation

## 🌱 Future Improvements

- Multilingual support
- Sentiment and emotion analysis
- Chatbot integration
- Real-time monitoring dashboard

## 🙏 Acknowledgements

- Twilio for SMS/voice API
- Scikit-learn and Imbalanced-learn
- The creators of the dataset

## 📬 Contact

Aswanikrishna [aswanikrishnac@gmail.com]
