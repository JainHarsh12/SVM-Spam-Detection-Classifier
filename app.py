import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample email dataset with spam and not-spam labels
data = {
    "Email": [
        "Hello, I am a prince and need your help", 
        "Meeting at 10 am tomorrow", 
        "Win a brand new car now! Click here", 
        "Don’t miss out on this limited-time offer",
        "Project deadline is approaching, please submit your report",
        "Congratulations, you have won a prize!",
        "Important: Your account has been compromised",
        "Free tickets available for a limited time only"
    ],
    "Label": ["Spam", "Not Spam", "Spam", "Spam", "Not Spam", "Spam", "Not Spam", "Spam"]
}
df = pd.DataFrame(data)

# Preprocess data
X = df["Email"]
y = df["Label"]

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Stratified train-test split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the SVM model
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = svm.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
