from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Email Spam Classifier API",
    description="Enter a message and classify it as Spam or Not Spam",
    version="1.0"
)

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

import pandas as pd
import nlp_myprocessor as mp

def preprocess_text_pipeline(text: str) -> str:
    # Wrap the text in a temporary DataFrame
    temp_df = pd.DataFrame({"text": [text]})

    # Apply all preprocessing steps
    mp.lowercase(temp_df, "text")
    mp.remove_stopwords(temp_df, "text")
    mp.remove_punctuation(temp_df, "text")
    mp.remove_emojis(temp_df,"text")
    mp.remove_urls(temp_df,"text")
    mp.remove_tags(temp_df,"text")

    # Return the processed text
    return temp_df["text"].iloc[0]


# Request model
class EmailInput(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Classifier API!"}

@app.post("/predict/")
def predict_spam(data: EmailInput):
    message = preprocess_text_pipeline(data.message)  # preprocess it

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]

    return {
        "input": data.message,
        "processed_input": message,
        "prediction": "Spam" if prediction == 1 else "Not Spam"}
