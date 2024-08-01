#textblob library
#create a sample text 
from textblob import TextBlob
#
texts =[
    "I love NLP ! It's works great and I'm very satisfied",
    "This is my first experience on doing sentiment analysis", "I am little bit disappointed"
    "The NLP sentiment analysis is quiet intereseting it is neither good or bad",
]

# create function to do do the sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    #-1.0-1.0 polarity score
    polarity = analysis.sentiment.polarity
    if polarity>0:
        sentiment="Positive"
    elif polarity<0:
        sentiment="negative"
    else:
        sentiment="neutral"
    return sentiment

for text in texts :
    sentiment = analyze_sentiment(text)
    print(f"Text:{text}")
    print(f"sentiment:{sentiment}\n")
    