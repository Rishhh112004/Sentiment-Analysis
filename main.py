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
