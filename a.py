import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# App title
# -----------------------------
st.title("📧 Spam Mail Detection App")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_data
def load_and_train_model():
    # Load dataset (make sure mail_data.csv is in the same folder as a.py)
    raw_mail_data = pd.read_csv("mail_data.csv")

    # Label encoding
    raw_mail_data.replace({'Category': {'ham': 0, 'spam': 1}}, inplace=True)

    # Split data
    X = raw_mail_data['Message']
    Y = raw_mail_data['Category']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Feature extraction
    vectorizer = TfidfVectorizer(
        min_df=1,
        stop_words='english',
        lowercase=True
    )

    X_train_features = vectorizer.fit_transform(X_train)
    Y_train = Y_train.astype('int')

    # Train model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    return model, vectorizer

model, feature_extraction = load_and_train_model()

# -----------------------------
# User input
# -----------------------------
user_input = st.text_area(
    "✍️ Enter the email message:",
    height=150
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Mail"):
    if user_input.strip() == "":
        st.warning("Please enter an email message.")
    else:
        input_mail = [user_input]
        input_mail_features = feature_extraction.transform(input_mail)
        prediction = model.predict(input_mail_features)

        if prediction[0] == 1:
            st.error("🚨 This is a **Spam Mail**")
        else:
            st.success("✅ This is a **Ham Mail**")
