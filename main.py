from textblob import TextBlob

print("Welcome to the Sentiment Analyzer!")
text = input("Enter a sentence: ")

blob = TextBlob(text)
sentiment = blob.sentiment.polarity

if sentiment > 0:
    print("😊 Positive")
elif sentiment < 0:
    print("😠 Negative")
else:
    print("😐 Neutral")


####################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import joblib
import string

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv('imdb.csv')

# Basic cleanup
df['review'] = df['review'].str.lower()
df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Features and labels
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # convert to 1/0

# Text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
