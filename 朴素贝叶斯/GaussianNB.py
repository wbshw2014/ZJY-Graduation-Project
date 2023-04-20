# -*- codeing = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 导入数据集
iris = load_iris()

# 将数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# 创建高斯朴素贝叶斯分类器
clf = GaussianNB()

# 训练高斯朴素贝叶斯分类器，并记录训练和验证集上的准确率
train_accs, val_accs = [], []

clf.fit(X_train, y_train)

train_pred = clf.predict(X_train)
val_pred = clf.predict(X_val)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

train_accs.append(train_acc)
val_accs.append(val_acc)

print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# 对于分类算法还可以观察precision，recall，f1-score等
print("\nClassification Report:")
print(classification_report(y_val, val_pred))

#混淆矩阵
plt.figure()
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.ylabel('Actual label')  # x轴标题
plt.xlabel('Predicted label')  # y轴标题
plt.show()

# 绘制贝叶斯分布
plt.figure()
sns.kdeplot(X_train[y_train == 0, 0], fill=True, label="Class 0")
sns.kdeplot(X_train[y_train == 1, 0], fill=True, label="Class 1")
sns.kdeplot(X_train[y_train == 2, 0], fill=True, label="Class 2")
plt.xlabel("Feature 0")
plt.ylabel("Density")
plt.title("Feature 0 Distribution by Class")
plt.legend()
plt.show()

# 绘制函数
plt.figure()
x = np.linspace(-10, 10, 100)
y = np.exp(-x ** 2)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Example Function")
plt.show()

# 绘制置信区间
plt.figure()
x = np.linspace(-10, 10, 100)
y = np.exp(-x ** 2)
plt.fill_between(x, y, 0, where=(y > 0), interpolate=True, alpha=0.2)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Example Confidence Interval")
plt.show()
