<<<<<<< HEAD
# -*- codeing = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为pandas数据框
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 使用前两个特征作为自变量
X = df[['sepal length (cm)', 'sepal width (cm)']]
# 使用第三个特征作为因变量
y = df['petal length (cm)']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建线性回归模型并训练它
model = LinearRegression()
model.fit(X_train, y_train)

# 输出模型的截距和系数
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# 对测试集进行预测并计算均方误差和决定系数
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: %.2f' % mse)
print('Coefficient of determination: %.2f' % r2)

# 绘制数据和拟合曲线
fig, ax = plt.subplots()
ax.scatter(X_test.iloc[:, 0], y_test, label='Actual')
ax.scatter(X_test.iloc[:, 0], y_pred, color='lightgreen', label='Predicted')
model_X0 = np.linspace(np.min(X_test.iloc[:, 0]), np.max(X_test.iloc[:, 0]), 100)
model_X1 = np.linspace(np.min(X_test.iloc[:, 1]), np.max(X_test.iloc[:, 1]), 100)
y_model = model.predict(np.vstack((model_X0, model_X1)).T)
ax.plot(model_X0, y_model, c='red', label='Linear Regress', zorder=0)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.legend()
plt.show()
=======
# -*- codeing = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为pandas数据框
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 使用前两个特征作为自变量
X = df[['sepal length (cm)', 'sepal width (cm)']]
# 使用第三个特征作为因变量
y = df['petal length (cm)']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建线性回归模型并训练它
model = LinearRegression()
model.fit(X_train, y_train)

# 输出模型的截距和系数
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# 对测试集进行预测并计算均方误差和决定系数
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: %.2f' % mse)
print('Coefficient of determination: %.2f' % r2)

# 绘制数据和拟合曲线
fig, ax = plt.subplots()
ax.scatter(X_test.iloc[:, 0], y_test, label='Actual')
ax.scatter(X_test.iloc[:, 0], y_pred, color='lightgreen', label='Predicted')
model_X0 = np.linspace(np.min(X_test.iloc[:, 0]), np.max(X_test.iloc[:, 0]), 100)
model_X1 = np.linspace(np.min(X_test.iloc[:, 1]), np.max(X_test.iloc[:, 1]), 100)
y_model = model.predict(np.vstack((model_X0, model_X1)).T)
ax.plot(model_X0, y_model, c='red', label='Linear Regress', zorder=0)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.legend()
plt.show()
>>>>>>> origin/main
