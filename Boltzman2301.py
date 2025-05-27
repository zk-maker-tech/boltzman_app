import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit   

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取excel文件
file_path = '图7中2301工作面实测数据.xlsx'
dataframe = pd.read_excel(file_path)
t = dataframe.iloc[:, 1]          # 时间间隔
wt_original = dataframe.iloc[:, 10]           # 下沉值
wt = np.abs(wt_original)  # 转换为正值

# 鲁棒性检查
if not np.any(wt > 0):
    print("错误：wt 数据中无正值，可能需要检查数据或模型。")
    exit()
if np.any(np.isnan(wt)) or np.any(np.isinf(wt)):
    print("错误：wt 包含 NaN 或无穷大值。")
    exit()

########################################################################
# Boltzmann时间函数拟合
def Boltzmann(t, A, t0, B):
    return -A / (1 + np.exp((t - t0) / B)) + A

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 初始参数猜测
initial_guess = [np.max(wt), np.median(t), np.std(t)]

# 使用 curve_fit 进行拟合
params, _ = curve_fit(Boltzmann, t, wt, p0=initial_guess)

# 获取拟合后的参数
A, t0, B = params

print(f"拟合得到的参数: A = {A}, t0 = {t0}, B = {B}")
r2 = calculate_r2(wt, Boltzmann(t, A, t0, B))
print(f"相关系数: r2 = {r2}")

Wm = np.max(wt)  # 使用最大下沉值
print(f"实测下沉值: Wm = {Wm}")
relative_error = (A - Wm) / Wm if Wm != 0 else np.nan  # 防止除零
print(f"相对误差: relative_error = {relative_error}")

# 绘图
plt.figure()
plt.plot(t, wt, c='r', label='实测数据')
plt.plot(t, Boltzmann(t, A, t0, B), c='b', label=f'Boltzmann拟合\nA={A:.2f}, t0={t0:.2f}, B={B:.2f}\nR²={r2:.3f}')
plt.xlabel("时间间隔 (t/d)")
plt.ylabel("下沉值 (wt)")
plt.title("Boltzmann 时间函数拟合")
plt.legend()
plt.grid(True)
plt.show()