# -*- codeing = utf-8 -*-
import pandas as pd
import xlrd

df = pd.read_excel('初测试.xlsx',header=None)

data = df.values[1::,0:-1]
target = df.values[1::, -1]

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=42)


