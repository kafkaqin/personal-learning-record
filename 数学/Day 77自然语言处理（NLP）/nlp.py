from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


categories = ['alt.atheism','soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='all', categories=categories,shuffle=True,random_state=42)
X = newsgroups.data
y = newsgroups.target
vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=42)

clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("准确率: ",clf.score(X_test,y_test))

print("\n分类报告:")
print(classification_report(y_test,y_pred,target_names=newsgroups.target_names))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=newsgroups.target_names,yticklabels=newsgroups.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('nlP.png')