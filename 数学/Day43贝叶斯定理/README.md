贝叶斯定理是用于更新在给定新证据或数据的情况下某假设的概率。其基本形式为：

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

这里：
- \( P(A|B) \) 是在事件 B 发生的条件下事件 A 发生的概率（后验概率）。
- \( P(B|A) \) 是在事件 A 发生的条件下事件 B 发生的概率（似然度）。
- \( P(A) \) 是事件 A 发生的概率（先验概率）。
- \( P(B) \) 是事件 B 发生的概率（边缘似然度）。

### 示例：疾病检测

假设有如下信息：
- 某种疾病的普遍率（先验概率）\( P(Disease) = 0.01 \)，即人群中1%的人患有此病。
- 如果一个人确实患有这种疾病，那么检测结果呈阳性的概率（真阳性率）\( P(Positive|Disease) = 0.99 \)。
- 如果一个人没有患这种疾病，检测结果仍可能呈阳性的概率（假阳性率）\( P(Positive|\neg Disease) = 0.05 \)。

现在的问题是，如果一个人检测结果呈阳性，他实际上患病的概率是多少？

#### 解答过程：

1. **计算先验概率**：\( P(Disease) = 0.01 \)
2. **计算似然度**：对于患病者检测呈阳性的概率 \( P(Positive|Disease) = 0.99 \)；对于非患病者检测呈阳性的概率 \( P(Positive|\neg Disease) = 0.05 \)。
3. **计算边缘似然度** \( P(Positive) \)：这是所有情况下检测呈阳性的总概率，可以通过全概率公式得出：
   \[ P(Positive) = P(Positive|Disease) \cdot P(Disease) + P(Positive|\neg Disease) \cdot P(\neg Disease) \]
   \[ P(Positive) = (0.99 \times 0.01) + (0.05 \times 0.99) \]
   \[ P(Positive) = 0.0099 + 0.0495 = 0.0594 \]

4. **应用贝叶斯定理计算后验概率** \( P(Disease|Positive) \)：
   \[ P(Disease|Positive) = \frac{P(Positive|Disease) \cdot P(Disease)}{P(Positive)} \]
   \[ P(Disease|Positive) = \frac{0.99 \times 0.01}{0.0594} \]
   \[ P(Disease|Positive) ≈ 0.1667 \]

这意味着即使测试结果呈阳性，这个人实际患病的概率大约是16.67%。这说明了即使检测方法相对准确，由于疾病的低普遍率和假阳性率的存在，阳性结果并不绝对意味着患病。这也强调了为什么需要考虑多次测试或其他诊断手段来确认初步检测的结果。