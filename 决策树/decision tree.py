# -*- codeing = utf-8 -*-
# 标准库
import numpy as np
import pandas as pd

# 可视化的库
import matplotlib.pyplot as plt
import seaborn as sns

# 建模和机器学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.model_selection import train_test_split

## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV


def static_database(dataframe):

    print('\n数据集前五行:\n',dataframe.head())  # 查看数据集前五行

    print('\n数据集大小:', dataframe.shape)

    # 对标签进行排序并且做出柱状图
    dataframe.iloc[:,0].value_counts().sort_index().plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title('Class distribution for the MNIST Dataset', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)

df = pd.read_csv('H:\pycharm\我的程序\数据集\mnist_csv\mnist_train.csv', header=None)# 导入训练集

# static_database(df)

X = df.iloc[:,1:]  # 选择所有行和列，但排除列1
y = df.iloc[:,0]  # 将label列作为预测值

def shape_database():
    # 将训练集与验证集的尺度进行输出
    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of X_valid:', X_valid.shape)
    print('Shape of y_valid:', y_valid.shape)

# 划分数据集
X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size = 0.3,random_state=0)
# shape_database()

#训练模型
dtModel = DecisionTreeClassifier()  # 建立模型
dtModel.fit(X_train,y_train)

#预测数据
prediction = dtModel.predict(X_valid)

#计算准确率
acc = accuracy_score(y_valid,prediction)
print(f"Sum Axis-1 as Classification accuracy: {acc* 100}")

# print ("Accuracy on Training Data：{0:.2f}%".format(dtModel.score(X_train, y_train)*100))
print ("Accuracy on Test Data：{0:.2f}%".format(dtModel.score(X_valid, y_valid)*100))

def confu_matrix():
    #混淆矩阵
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_valid,  prediction)

    ax = sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
    plt.ylabel('Actual label')  # x轴标题
    plt.xlabel('Predicted label')  # y轴标题

# confu_matrix()

def more_Evaluate():
    # 对于分类算法还可以观察accuracy，Precision，Recall，F-score等
    print("\nClassification Report:")
    print(classification_report(y_valid, prediction))

# more_Evaluate()


def print_testimage(index):
    df1 = pd.read_csv('H:\pycharm\我的程序\数据集\mnist_csv\mnist_test.csv', header=None)  # 导入测试集

    some_digit = df1.iloc[index, 1:].values  # 按照索引取出图片
    some_digit_img = some_digit.reshape(28,28)
    plt.imshow(some_digit_img,'binary')  # 展示

# print_testimage(2000) # 查看第2000张图片

def predict_pic():
    figr,axes=plt.subplots(figsize=(10,10),ncols=3,nrows=3)
    axes=axes.flatten()
    for i in range(0,9):  # 循环
        jj=np.random.randint(0,X_valid.shape[0])          #挑选一个随机图片
        axes[i].imshow(X_valid.iloc[[jj]].values.reshape(28,28))
        axes[i].set_title('predicted: '+str(dtModel.predict(X_valid.iloc[[jj]])[0]))

# predict_pic()


def hyperparam():
    ## 定义参数取值范围

    parameters = {'splitter':('best','random')
                  ,'criterion':("gini","entropy")
                  ,"max_depth":[*range(1,10)]
                  ,'min_samples_leaf':[*range(1,50,1)]
    }
    model = DecisionTreeClassifier()

    ## 进行网格搜索
    clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=3, n_jobs=-1)
    clf = clf.fit(X_train, y_train)

    print(clf.best_params_)   # 得到最好的参数

    # 使用最好的参数再一次输出
    model = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=1, splitter='best')
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    acc = accuracy_score(y_valid, prediction)
    print(f"Sum Axis-1 as Classification accuracy: {acc * 100}")

    return clf

# hyperparam()