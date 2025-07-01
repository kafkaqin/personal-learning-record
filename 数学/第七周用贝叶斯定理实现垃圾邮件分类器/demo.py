from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd

data = pd.DataFrame({
    'text': [
        "Free Viagra now !!!",
        "Hi Bob,how about a game of golf tomorrow?",
        "Life Insurance for free",
        "Important meeting at 10am tomorrow!"
    ],
    'label': [1,0,1,0]
})

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'],test_size=0.25, random_state=42)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print("准确率:",accuracy_score(y_test, predicted))
print("分类报告:\n",classification_report(y_test, predicted))

new_email = ["Win money now!","Hello how are you?"]
predictions = text_clf.predict(new_email)
for email,prediction in zip(new_email,predictions):
    print(f"邮件\"{email}\" 被分类为 {'垃圾邮件'if prediction==1 else '正常邮件'}")