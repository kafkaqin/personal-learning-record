下面是一个完整的示例，展示如何使用 **TF-IDF（Term Frequency-Inverse Document Frequency）** 对文本进行向量化，并使用简单的分类模型（如逻辑回归）进行**文本分类**。我们将使用 Python 的 `scikit-learn` 库，以经典的 `20 Newsgroups` 数据集为例进行演示。

---

## ✅ 步骤概览

1. 加载文本数据集
2. 使用 `TfidfVectorizer` 将文本转换为 TF-IDF 向量
3. 划分训练集和测试集
4. 训练分类模型（如逻辑回归）
5. 评估模型性能（准确率、混淆矩阵等）

---

## 🧰 所需库安装（如果未安装）

```bash
pip install scikit-learn matplotlib seaborn
```

---

## 📊 示例代码：TF-IDF + 文本分类

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据集（只选两个类别，简化示例）
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

X = newsgroups.data  # 原始文本数据
y = newsgroups.target  # 标签（0 或 1）

# 2. 使用 TF-IDF 向量化文本
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 4. 训练模型（逻辑回归）
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 5. 预测与评估
y_pred = clf.predict(X_test)

# 准确率
print("准确率：", clf.score(X_test, y_test))

# 分类报告
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups.target_names, yticklabels=newsgroups.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()
```

---

## 📋 示例输出（类似如下）：

```
准确率： 0.92

分类报告：
                     precision    recall  f1-score   support

       alt.atheism       0.92      0.93      0.92        72
soc.religion.christian       0.93      0.91      0.92        80

           accuracy                           0.92       152
          macro avg       0.92      0.92      0.92       152
       weighted avg       0.92      0.92      0.92       152
```

---

## 🧠 简要解释

### 什么是 TF-IDF？
- **TF（Term Frequency）**：一个词在文档中出现的频率。
- **IDF（Inverse Document Frequency）**：衡量一个词在整个语料库中的普遍性。越常见的词 IDF 越低。
- TF-IDF 综合了这两个指标，能有效突出在文档中重要、在语料库中少见的词语。

### 为什么使用 TF-IDF？
- 相比于简单的词袋模型（Bag-of-Words），TF-IDF 更能体现词语的重要性。
- 在文本分类、信息检索、搜索引擎中广泛应用。

---

## 📚 补充说明

| 技术 | 说明 |
|------|------|
| `TfidfVectorizer` | 可设置 `ngram_range=(1,2)` 来考虑双词组合 |
| `stop_words='english'` | 去除常见无意义的英文停用词 |
| 分类器 | 可替换为 SVM、随机森林、朴素贝叶斯等 |
| 多分类任务 | 可扩展为多个类别，只需去掉 `categories` 参数限制 |

---

## ✅ 小贴士

- 如果你有自己的文本数据（CSV 文件），可以使用 `pandas` 读取后替换 `X` 和 `y`。
- 对中文文本，需要先进行分词处理（如 jieba），再进行 TF-IDF 向量化。
- 可使用 `GridSearchCV` 进行超参数调优。