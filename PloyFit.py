import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

data = pd.read_excel(r"C:\Users\86178\Desktop\项目组\new\data\预测数据_LSTM.xls")

x_original = data.values[:, 0]
# x_original = np.reshape(x_original, (x_original.shape[0], 1))
# x_original = MinMaxScaler().fit_transform(x_original)
# x_original = x_original.reshape(-1)

x_predict = data.values[:, 3]

target_original = data.values[:, 4]
# target_original = np.reshape(target_original, (target_original.shape[0], 1))
# target_original = MinMaxScaler().fit_transform(target_original)
# target_original = target_original.reshape(-1)
# target_predict = data.values[:, 3]

model = []
for i in range(len(x_predict) // 10):
    n = 4
    if i == 0:
        temp_x = x_predict[0:6]
        temp_y = target_original[0:6]
    else:
        temp_x = x_predict[i * 10 - 4:(i + 1) * 10 - 4]
        temp_y = target_original[i * 10 - 4:(i + 1) * 10 - 4]
    a = np.polyfit(temp_x, temp_y, n)
    b = np.poly1d(a)
    model.append(b)

nihe = []
for i in range(len(x_predict) // 10):
    if i == 0:
        temp_x = x_predict[0:6]
    else:
        temp_x = x_predict[i * 10 - 4:(i + 1) * 10 - 4]
    data = model[i](temp_x)
    for i in data:
        nihe.append(i)
nihe = np.array(nihe)
nihe = np.reshape(nihe, (nihe.shape[0], 1))
nihe = MinMaxScaler().fit_transform(nihe)
nihe = nihe.reshape(-1)

# temp_x = x_original[100:]
# temp_y = target_original[100:]
# a = np.polyfit(temp_x, temp_y, 3)
# b = np.poly1d(a)
# nihe = b(temp_x)

plt.plot(nihe, 'r', label='拟合值')
plt.plot(target_original, 'b', label='原始值')
plt.legend()
plt.show()
