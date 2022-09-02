import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from keras import callbacks
import matplotlib.pyplot as plt
from Export_Data import Export_Data


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class LSTM_Model(object):
    step = 4  # 时间步长

    def __init__(self):
        self.step = 4

    def get_data(self, data_name, dim):
        df = pd.read_excel(data_name)
        values = df.values
        # print("values:",values.shape)
        values = values[:, dim]
        # print(values[2])
        values = np.reshape(values, (values.shape[0], 1))
        values = MinMaxScaler().fit_transform(values)
        return values

    # split a univariate sequence into samples
    def split_sequence(self, data, seq_len):
        data = pd.DataFrame(data)
        amount_of_features = len(data.columns)
        data = data.iloc[:, :].values  # pd.DataFrame(stock)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length + 1):
            result.append(data[index: index + sequence_length])
        result = np.array(result)
        train = result[:, :]
        x_train = train[:, :-1]
        y_train = train[:, -1][:, -1]
        return x_train, y_train

    def train_process(self):
        df1 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\7_breach_temp.xlsx", 2)
        df2 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\8_breach_temp.xlsx", 1)
        # df3 = self.get_data("C:\\Users\\TITAN3\\Desktop\\核电站系统流程框架\\温度数据\\8_breach_temp.xlsx", 4)
        df3 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\小破口1.0_temp.xlsx", 1)
        df = np.concatenate((df1, df2, df3), axis=0)
        # df = np.concatenate((df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15), axis=0)
        print(df.shape)
        X_train, y_train = self.split_sequence(df, self.step)
        print(X_train.shape)

        return X_train, y_train

    def build_model(self):  # (1, 5, 1)
        d = 0.2
        model = Sequential()
        # model.add()
        model.add(LSTM(128, activation='tanh', input_shape=(self.step, 1), return_sequences=True))
        # model.add(SimpleRNN(128, activation='tanh',input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(self.step, 1), return_sequences=False))
        model.add(Dropout(d))
        # model.add(LSTM(32,input_shape=(layers[1], layers[0]), return_sequences=False))
        # model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse', optimizer='Adam')
        print(model.summary())

        return model

    def train(self, model, xTrain, yTrain):
        history = LossHistory()
        model.fit(
            xTrain,
            yTrain,
            batch_size=32,
            epochs=50,
            validation_split=0.01,
            verbose=1,
            callbacks=[history])
        return model, history

    def predict(self, model):
        df_test = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\7_breach_temp.xlsx", 1)
        print(df_test.shape)
        X_TEST, Y_TEST = self.split_sequence(df_test[:], self.step)
        # print(X_TEST.shape, Y_TEST.shape)
        yhat = model.predict(X_TEST)
        # print(yhat.shape)
        loss_ave = model.evaluate(X_TEST, Y_TEST)
        print(model.evaluate(X_TEST, Y_TEST))

        return Y_TEST, yhat, loss_ave

    def show(self, Y_TEST, yhat, loss_ave):
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12

        # 画预测值与原始值
        plt.plot(yhat, color='darkorange', label='Simulated Value', linestyle='-.', marker='^', markevery=10)
        plt.plot(Y_TEST, color='lightseagreen', label='Original Value', marker='*', markevery=10)
        # plt.plot(x, Y_TEST, color='blue', label='original value')

        plt.title("7_breach_SG_Break")
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized SG_Break Temperature')
        plt.legend(loc='upper right')
        plt.text(50, 0.8, 'Loss:' + "{:.6f}".format(loss_ave))
        plt.grid()  # 添加网格
        plt.axis([0, 150, 0.0, 1.3])
        plt.show()


def main():
    lstm = LSTM_Model()
    model = lstm.build_model()
    X_train, Y_train = lstm.train_process()
    lstm.train(model, X_train, Y_train)
    Y_TEST, yhat, loss_ave = lstm.predict(model)
    lstm.show(Y_TEST, yhat, loss_ave)
    Export_Data(Y_TEST.tolist(), yhat.tolist(), 0)


if __name__ == '__main__':
    main()
