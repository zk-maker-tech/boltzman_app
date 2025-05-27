import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 文件并验证
file_path = '图12实验数据.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件 {file_path} 不存在")
dataframe = pd.read_excel(file_path)
if dataframe.shape[1] < 9:
    raise ValueError("Excel 文件必须至少包含9列")
x = dataframe.iloc[:, 7]  # 时间间隔
y = dataframe.iloc[:, 8]
if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
    raise ValueError("数据中包含无效值（NaN 或无穷大）")


# 线性拟合函数（支持带截距）
def linear_fit(x, y, with_intercept=False):
    if with_intercept:
        # 带截距的拟合：y = kx + b
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator_k = np.sum(x * y) - n * x_mean * y_mean
        denominator_k = np.sum(x ** 2) - n * x_mean ** 2
        if denominator_k < 1e-9:
            raise ValueError("分母接近 0，无法计算斜率 k")
        k = numerator_k / denominator_k
        b = y_mean - k * x_mean
        y_fit = k * x + b
    else:
        # 无截距的拟合：y = kx
        numerator = np.sum(x * y)
        denominator = np.sum(x ** 2)
        if denominator < 1e-9:
            raise ValueError("分母接近 0，无法计算斜率 k")
        k = numerator / denominator
        b = 0
        y_fit = k * x

    # 计算 R^2
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_fit) ** 2)
    if ss_tot < 1e-9:
        r_squared = 1.0 if ss_res < 1e-9 else 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return k, b, y_fit, r_squared


# 执行拟合（可选择是否带截距）
k, b, y_fit, r_squared = linear_fit(x, y, with_intercept=False)

# 打印结果
if b == 0:
    print(f"拟合得到的斜率 k = {k:.4f}")
else:
    print(f"拟合得到的斜率 k = {k:.4f}, 截距 b = {b:.4f}")
print(f"决定系数 R^2 = {r_squared:.4f}")

# 绘制原始数据点和拟合直线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='原始数据')
if b == 0:
    plt.plot(x, y_fit, color='red', label=f'拟合直线 y = {k:.2f}x')
else:
    plt.plot(x, y_fit, color='red', label=f'拟合直线 y = {k:.2f}x + {b:.2f}')
plt.xlabel('时间间隔 (s)')
plt.ylabel('测量值')
plt.title(f'线性拟合结果 (R^2 = {r_squared:.4f})')
plt.legend(loc='best')
plt.grid(True)
plt.xlim(min(x) * 0.9, max(x) * 1.1)
plt.ylim(min(y) * 0.9, max(y) * 1.1)
plt.show()