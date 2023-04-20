import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, classification_report
import graphviz

# 导入数据集
iris = load_iris()

# 将数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练决策树分类器，并记录训练和验证集上的准确率和损失
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for i in range(1, 11):
    # 限制决策树的最大深度
    clf = DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_loss = log_loss(y_train, clf.predict_proba(X_train))
    val_loss = log_loss(y_val, clf.predict_proba(X_val))

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch: {i}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 对于分类算法还可以观察accuracy，Precision，Recall，F-score等
print("\nClassification Report:")
print(classification_report(y_val, val_pred))

#混淆矩阵
plt.figure()
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.ylabel('Actual label')  # x轴标题
plt.xlabel('Predicted label')  # y轴标题

# 可视化训练和验证集的准确率和损失
epochs = range(1, 11)

plt.figure()
plt.plot(epochs, train_accs, 'b', label='Training Accuracy')
plt.plot(epochs, val_accs, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, train_losses, 'b', label='Training Loss')
plt.plot(epochs, val_losses, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# generate visualization of decision tree
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=None,
                           class_names=None,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")
