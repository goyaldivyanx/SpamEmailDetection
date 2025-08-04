import numpy as np
import pandas as pd
import nlp_myprocessor as mp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib

df = pd.read_csv(r"D:\NLP\EmailClassifier\EmailClassifierDataset\combined_data.csv")
mp.lowercase(df,"text")
mp.remove_stopwords(df,"text")
mp.remove_punctuation(df,"text")
df.dropna(inplace=True)

#print(df.head())

x_train,x_test,y_train,y_test = train_test_split(df["text"],df["label"],test_size=0.2,random_state=24)

cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

model = MultinomialNB()
model.fit(x_train,y_train)


y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(cv, "count_vectorizer.pkl")