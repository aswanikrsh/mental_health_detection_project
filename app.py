import os
from flask import Flask, render_template, request
import pickle
from twilio.rest import Client

app = Flask(__name__)

# Load model and vectorizer
with open("mental_health_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Twilio credentials (replace with actual values)
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = '+1234567890'  # Your Twilio number
EMERGENCY_PHONE_NUMBER = '+91xxxxxxxxxx'  # Person to alert

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Send SMS
def send_sms_alert():
    try:
        message = client.messages.create(
            body="🚨 Mental Health Alert: A user may be showing suicidal signs. Please provide immediate support.",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_PHONE_NUMBER
        )
        print("SMS sent:", message.sid)
    except Exception as e:
        print("SMS sending failed:", e)

# Make voice call
def make_voice_call():
    try:
        call = client.calls.create(
            twiml='<Response><Say>This is a mental health alert. A user may need urgent help. Please take immediate action.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_PHONE_NUMBER
        )
        print("Call initiated:", call.sid)
    except Exception as e:
        print("Call failed:", e)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, risk=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"].strip()

    if not text:
        return render_template("index.html", prediction="❌ No input provided", risk="none")

    transformed_text = tfidf_vectorizer.transform([text])
    prediction_numeric = model.predict(transformed_text)[0]

    label_map = {0: "safe", 1: "suicide"}
    prediction = label_map.get(prediction_numeric, "unknown")

    confidence = model.predict_proba(transformed_text).max()

    if confidence < 0.6:
        return render_template("index.html", prediction="⚠️ Not sure. Try rephrasing.", risk="none")

    if prediction == "suicide":
        send_sms_alert()
        make_voice_call()

    risk = "high" if prediction == "suicide" else "none"
    return render_template("index.html", prediction=prediction, risk=risk)

# ✅ Help route for "Get Help" button
@app.route("/help")
def help_page():
    return render_template("help.html")

if __name__ == "__main__":
    app.run(debug=True)







