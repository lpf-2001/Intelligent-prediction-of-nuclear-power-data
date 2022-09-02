import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(r"C:\Users\86178\Desktop\项目组\new\data\7_breach_temp.xlsx")
values = df.values
values = values[1:, 1]
# print(values)
#归一化
values = np.reshape(values, (values.shape[0], 1))
values = MinMaxScaler().fit_transform(values)
# print(values)
plt.figure()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15
plt.plot(values, color='blue', label='Original Value')
plt.xlabel('Time (s)')
plt.ylabel('Normalized SG Temperature')
plt.axis([0, 150, 0.0, 1.1])
#plt.text(80, 1.0, 'Break Size:0.2$\mathregular{cm^2}$')
#plt.text(80, 0.9, 'Power:100%')
# plt.text(80, 0.8, 'Loss:' + str(0.00345))
plt.grid()  # 添加网格
plt.show()
