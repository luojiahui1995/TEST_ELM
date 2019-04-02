import numpy as np
from sklearn import metrics

def to3(num_data,y_train,classes):

    # 将只有一列的Label矩阵转换，例如，iris的label中共有三个值，则转换为3列，以行为单位，label值对应位置标记为1，其它位置标记为0
    train_y = np.zeros((num_data, classes))  # 初始化一个120行，3列的全0矩阵
    for i in range(0, num_data):
        train_y[i, y_train[i]] = 1  # 对应位置标记为1
    return train_y

def result_max(pred_Y):
   # 取输出节点中值最大的类别作为预测值  (将3列转为一列,pred_Y是train的输出)
    predy = []
    for i in pred_Y:
       L = i.tolist()
       predy.append(L.index(max(L)))
    return predy

def score(y_test,pred_y ):
    print("准确率：")
    # 使用准确率方法验证
    print(metrics.accuracy_score(y_true=y_test, y_pred=pred_y))