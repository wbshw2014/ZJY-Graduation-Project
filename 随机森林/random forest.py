<<<<<<< HEAD
# -*- codeing = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 导入数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练随机森林分类器
clf.fit(X_train, y_train)

# 预测测试集的结果
y_pred = clf.predict(X_test)

# 输出模型的准确率
print('Accuracy:', accuracy_score(y_test, y_pred))

# 对于分类算法还可以观察accuracy，Precision，Recall，F-score等
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#混淆矩阵
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.ylabel('Actual label')  # x轴标题
plt.xlabel('Predicted label')  # y轴标题
plt.show()
=======
# -*- codeing = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 导入数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练随机森林分类器
clf.fit(X_train, y_train)

# 预测测试集的结果
y_pred = clf.predict(X_test)

# 输出模型的准确率
print('Accuracy:', accuracy_score(y_test, y_pred))

# 对于分类算法还可以观察accuracy，Precision，Recall，F-score等
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#混淆矩阵
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.ylabel('Actual label')  # x轴标题
plt.xlabel('Predicted label')  # y轴标题
plt.show()
>>>>>>> origin/main
