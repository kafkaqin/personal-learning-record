from textblob import TextBlob
import pandas as pd


data = [
    "I love the new design of the website!",
    "This product is terrible,I regret buying it.",
    "The customer service here is amazing.",
    "Absolutely awful experience with this company"
]

df = pd.DataFrame(data, columns=['text'])

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['polarity'] = df['text'].apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(classify_sentiment)

print(df)


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

data = [
    "I love the new design of the website!",
    "This product is terrible,I regret buying it.",
    "The customer service here is amazing.",
    "Absolutely awful experience with this company"
]

df = pd.DataFrame(data, columns=['text'])

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs

df['vader_score'] = df['text'].apply(get_vader_sentiment)

def classify_vader_sentiment(score):
    compound = score['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['vader_sentiment'] = df['vader_score'].apply(lambda x: classify_vader_sentiment(x))
print(df)

from snownlp import SnowNLP
import pandas as pd

data = ["这款手机真的很好看,拍照也很清晰.",
        "太失望了，根本不好用.",
        "客服态度非常差，不会再来买了.",
        "服务很好，下次还会光顾。"]

df = pd.DataFrame(data, columns=['text'])

def get_chinese_sentiment(text):
    analysis = SnowNLP(text)
    return analysis.sentiments

df['sentiment_score'] = df['text'].apply(get_chinese_sentiment)

def classify_chinese_sentiment(score):
    if score >= 0.6:
        return 'Positive'
    elif score <= 0.4:
        return 'Negative'
    else:
        return 'Neutral'
df['sentiment'] = df['sentiment_score'].apply(classify_chinese_sentiment)
print(df)