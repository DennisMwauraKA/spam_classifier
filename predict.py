import joblib
import re


model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("spam_vectorizer.joblib")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Step 3: Loop to test user messages
print("SMS Spam Detector")
print("Type your message below (type 'exit' to quit):\n")

while True:
    msg = input("Your message: ")
    if msg.lower() == "exit":
        print("Goodbye!")
        break

    # Clean and vectorize
    cleaned = clean_text(msg)
    msg_vec = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(msg_vec)[0]

    if prediction == 1:
        print(" Prediction: SPAM")
    else:
        print(" Prediction: HAM (not spam)")
