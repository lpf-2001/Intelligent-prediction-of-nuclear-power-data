# univariate convlstm example
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from sklearn.preprocessing import MinMaxScaler
from keras import callbacks
import matplotlib.pyplot as plt



class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ConvLSTM(object):

    def __init__(self):
        self.n_features = 1
        self.n_seq = 2
        self.n_step = 2

    def get_data(self, data_name, dim):
        df = pd.read_excel(data_name)
        values = df.values
        # print("values:",values.shape)
        values = values[:, dim]
        # print(values[2])
        # values = values[1:-5, 35:39]
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
        # print("原始数据形状为：",data.shape)
        for index in range(len(data) - sequence_length + 1):
            result.append(data[index: index + sequence_length])
        result = np.array(result)
        result = result[:, :]
        x = result[:, :-1]
        y = result[:, -1][:, -1]
        return x, y

    # define input sequence
    def train_process(self):
        df1 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\7_breach_temp.xlsx", 2)
        df2 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\8_breach_temp.xlsx", 1)
        # df3 = self.get_data("C:\\Users\\TITAN3\\Desktop\\核电站系统流程框架\\温度数据\\8_breach_temp.xlsx", 4)
        df3 = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\小破口1.0_temp.xlsx", 1)
        df = np.concatenate((df1, df2, df3), axis=0)

        # choose a number of time steps
        n_steps = 4
        # split into samples
        X, y = self.split_sequence(df, n_steps)
        print(X.shape)
        print(y.shape)

        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        X = X.reshape((X.shape[0], self.n_seq, 1, self.n_step, self.n_features))
        print(X.shape)
        return X, y

    # define model
    def build_model(self):
        model = Sequential()
        model.add(
            ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                       input_shape=(self.n_seq, 1, self.n_step, self.n_features)))
        model.add(Flatten())
        # model.add(RepeatVector(1))
        # model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    # fit model
    def train(self, model, X_train, y_train):
        history = LossHistory()
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.01, verbose=1, callbacks=[history])

    # demonstrate prediction
    def predict(self, model):
        df_test = self.get_data(r"C:\Users\86178\Desktop\项目组\new\data\7_breach_temp.xlsx", 1)
        n_steps = 4
        X_TEST, Y_TEST = self.split_sequence(df_test, n_steps)
        X_TEST = X_TEST.reshape((X_TEST.shape[0], self.n_seq, 1, self.n_step, self.n_features))

        yhat = model.predict(X_TEST)
        loss_ave = model.evaluate(X_TEST, Y_TEST)
        print(yhat.shape, Y_TEST.shape)

        return Y_TEST, yhat, loss_ave

    def show(self, Y_TEST, yhat, loss_ave):
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12

        #画预测值与原始值
        plt.plot(yhat, color='darkorange', label='Simulated Value', linestyle='-.', marker='^', markevery=10)
        plt.plot(Y_TEST, color='lightseagreen', label='Original Value', marker='*', markevery=10)

        #x轴，y轴及标签
        plt.title("7_breach_Exit")
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Core Exit Temperature')
        plt.legend(loc='upper right')
        plt.text(50, 0.8, 'Loss:' + "{:.6f}".format(loss_ave))
        plt.grid()  # 添加网格
        plt.axis([0, 150, 0.0, 1.1])
        # plt.savefig("pressurizer pressure-0.2-100.jpg")
        # plt.savefig("core inlet temperature-0.2-100.jpg")
        plt.show()


def main():
    convlstm = ConvLSTM()
    model = convlstm.build_model()
    x_train, y_train = convlstm.train_process()
    convlstm.train(model, x_train, y_train)
    Y_TEST, yhat, loss_ave = convlstm.predict(model)
    convlstm.show(Y_TEST, yhat, loss_ave)


if __name__ == '__main__':
    main()
