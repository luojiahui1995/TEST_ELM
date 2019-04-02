import numpy as np
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
from sklearn import metrics # 引入包含数据验证方法的包
from publicfc import to3
from publicfc import result_max
from publicfc import score

class SingeHiddenLayer(object):

    def __init__(self, X, y, num_hidden):
        self.data_x = np.atleast_2d(X)# 判断输入训练集是否大于等于二维; 把x_train()取下来
        self.data_y = np.array(y).flatten()  # a.flatten()把a放在一维数组中，不写参数默认是“C”，也就是先行后列的方式，也有“F”先列后行的方式； 把 y_train取下来
        self.num_data = len(self.data_x)  # 训练数据个数
        self.num_feature = self.data_x.shape[1];  # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shape[1]==4)
        self.num_hidden = num_hidden;  # 隐藏层节点个数


        # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
        self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden))

        # 随机生成偏置，一个隐藏层节点对应一个偏置
        for i in range(self.num_hidden):
            b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
            self.first_b = b

        # 生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
        for i in range(self.num_data - 1):
            b = np.row_stack((b, self.first_b))  # row_stack 以叠加行的方式填充数组
        self.b = b


    # 定义sigmoid函数
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))



    def train(self, x_train, y_train, classes):
        mul = np.dot(self.data_x, self.w)  # 输入乘以权重
        add = mul + self.b  # 加偏置
        H = self.sigmoid(add)  # 激活函数

        H_ = np.linalg.pinv(H)  # 求广义逆矩阵
        # print(type(H_.shape))
        train_y = to3(self.num_data, y_train, classes)
        self.out_w = np.dot(H_, train_y)  # 求输出权重

        return self.out_w, train_y
    def predict_pre(self, x_test):
        self.t_data = np.atleast_2d(x_test)  # 测试数据集
        self.num_tdata = len(self.t_data)  # 测试集的样本数
        self.pred_Y = np.zeros((x_test.shape[0]))  # 初始化

        b = self.first_b

        # 扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
        for i in range(self.num_tdata - 1):
            b = np.row_stack((b, self.first_b))  # 以叠加行的方式填充数组
        return b

    def predict(self, x_test, b):
        # self.t_data = np.atleast_2d(x_test)# 测试数据集
        # self.num_tdata = len(self.t_data)  # 测试集的样本数
        # self.pred_Y = np.zeros((x_test.shape[0]))  # 初始化
        #
        # b = self.first_b
        #
        # # 扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
        # for i in range(self.num_tdata - 1):
        #     b = np.row_stack((b, self.first_b))  # 以叠加行的方式填充数组

        # 预测
        print(self.t_data.shape)
        print(self.w.shape)
        print(b.shape)
        self.pred_Y = np.dot(self.sigmoid(np.dot(self.t_data, self.w) + b), self.out_w)
        print(self.sigmoid(np.dot(self.t_data, self.w) + b).shape)
        print(self.out_w.shape)
        #print(np.dot(self.sigmoid(np.dot(self.t_data, self.w) + b), self.out_w))

        return self.pred_Y
    # def result_max(self,pred_Y):
    #     # 取输出节点中值最大的类别作为预测值  (将3列转为一列)
    #     self.predy = []
    #     for i in self.pred_Y:
    #         L = i.tolist()
    #         self.predy.append(L.index(max(L)))
    #     return predy

    # def score(self, y_test,):
    #     print("准确率：")
    #     # 使用准确率方法验证
    #     print(metrics.accuracy_score(y_true=y_test, y_pred=self.predy))


stdsc = StandardScaler()  # StandardScaler类,利用接口在训练集上计算均值和标准差，以便于在后续的测试集上进行相同的缩放
iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
x, y = stdsc.fit_transform(iris.data), iris.target  # 数据归一化
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



ELM = SingeHiddenLayer(x_train, y_train, 10)  # 训练数据集，训练集的label，隐藏层节点个数
out_w = ELM.train(x_train, y_train, 3)[0]
b1 = ELM.predict_pre(x_train)#使用测试集预测准备，给出偏置b
out_y = ELM.predict(x_train, b1)#训练集输入，用训练所得out_w求的y~
print(out_y)
train_y =ELM.train(x_train, y_train, 3)[1]
print(train_y)
e = train_y - out_y#求得残差e，e将作为下次计算的输入
b1 = ELM.predict_pre(x_test)
y1_pred = ELM.predict(x_test, b1)#第一层直接输出的预测结果3列
Y1_pred = result_max(y1_pred)
#print(Y1_pred)
score(y_test, Y1_pred)
ELM2 = SingeHiddenLayer(x_test, y_test, 4)
ELM2.train(e, y_train, 3)[0]#用第一层训练的残差e作为输入进行第二层训练，得到out_w2
b = ELM2.predict_pre(x_test)#预测准备，使用测试集。给出偏执b
out_Y =ELM.predict(x_test, b1) + ELM2.predict(x_test, b)
# out_Y = y1_pred + out_y2
# print(y1_pred)
# print("--------------------------------------------------------------")
# print(out_y2)
# print("--------------------------------------------------------------")
# print(out_Y)
pred_Y = result_max(out_Y)
# print(pred_Y)
# print(pred_Y)
score(y_test, pred_Y)




