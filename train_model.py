# ============================================
# Mental Health Risk Detection - Model Training
# ============================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================
# 1. LOAD DATASET
# ============================================

print("Loading dataset...")

df = pd.read_csv("final_merged_dataset.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("Columns:", df.columns)


# ============================================
# 2. CREATE INPUT TEXT COLUMN
# ============================================

print("\nCreating combined text column...")

df["text"] = df["post"].astype(str) + " " + df["question"].astype(str) + " " + df["reasoning"].astype(str)

X = df["text"]
y = df["stress_label"]


# ============================================
# 3. TRAIN TEST SPLIT
# ============================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 4. TEXT VECTORIZATION (TF-IDF)
# ============================================

print("\nVectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization complete")


# ============================================
# 5. TRAIN MODELS
# ============================================

print("\nTraining models...")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_vec, y_train)
rf_pred = rf_model.predict(X_test_vec)

# Support Vector Machine
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)


# ============================================
# 6. MODEL EVALUATION FUNCTION
# ============================================

def evaluate_model(name, y_test, predictions):

    print("\n===================================")
    print("MODEL:", name)
    print("===================================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ============================================
# 7. EVALUATE ALL MODELS
# ============================================

evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("SVM", y_test, svm_pred)


# ============================================
# 8. SAVE BEST MODEL
# ============================================

print("\nSaving best model...")

joblib.dump(lr_model, "stress_detection_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved successfully!")

print("\nFiles created:")
print("stress_detection_model.pkl")
print("tfidf_vectorizer.pkl")

print("\nTraining pipeline completed successfully!")