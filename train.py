import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import pickle

# Load and clean dataset
df = pd.read_csv("Suicide_Detection.csv")
df["class"] = df["class"].str.strip()
df = df.dropna(subset=["text", "class"])

# Map labels
y = df["class"].map({"non-suicide": 0, "suicide": 1})
X = df["text"]

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("🔍 Evaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred, target_names=["non-suicide", "suicide"]))

# Save model & vectorizer
with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved.")






