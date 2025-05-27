import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from scipy.optimize import least_squares # 当前未使用
from scipy.optimize import curve_fit, OptimizeWarning

# from scipy.optimize import leastsq # 当前未使用, curve_fit 通常是首选

# 优先尝试这些字体（按优先级排序）
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 微软雅黑（含更多特殊字符）
    'SimHei',  # 黑体
    'Arial Unicode MS',  # Mac系统通用字体
    'sans-serif'  # 兜底的无衬线字体
]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取excel文件
file_path = '图5文献22和30点实测数据.xlsx'
try:
    dataframe = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"错误：Excel文件 '{file_path}' 未找到。请检查文件名和路径。")
    exit()

# 提取数据并转换为NumPy数组
try:
    t = dataframe.iloc[:, 1].values  # 时间间隔 (确保为数值类型)
    wt_original = dataframe.iloc[:, 2].values  # 下沉值 (确保为数值类型)
except IndexError:
    print(f"错误：无法从Excel文件中正确提取数据列。请检查文件 '{file_path}' 的列结构。")
    print("期望第二列为时间(t)，第三列为下沉值(wt)。")
    exit()
except Exception as e:
    print(f"读取Excel数据时发生错误: {e}")
    exit()

wt = np.abs(wt_original)  # 通常下沉是负值，模型也设计为拟合这种趋势。
# 如果原始数据中wt就是正值代表下沉大小，abs()可能不需要。
# 这里保持abs()以确保wt为正，与后续模型输出正值对应。

# 获取数据的基本特征用于初始猜测和边界设定
if len(wt) == 0 or len(t) == 0:
    print("错误：时间(t)或下沉值(wt)数据为空。")
    exit()

A_observed_max = np.max(wt) if len(wt) > 0 else 1.0  # 观测到的最大下沉
t_observed_max = np.max(t) if len(t) > 0 else 1.0  # 观测到的最大时间
t_observed_median = np.median(t) if len(t) > 0 else 0.5 * t_observed_max  # 观测时间中位数


# ------------------------------------------------------------------------
# 辅助函数定义
# ------------------------------------------------------------------------
def calculate_rmse(y_true, y_pred):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def calculate_relative_error(y_true, y_pred):
    """计算相对误差 (%)，增加分母保护"""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # 避免除以零，对于真实值为零的点，相对误差没有良好定义，可以排除或特殊处理
    # 这里，如果真实值接近零，相对误差可能会非常大
    epsilon = 1e-9  # 一个小常数防止除以零
    valid_indices = np.abs(y_true_arr) > epsilon
    if not np.any(valid_indices):
        return np.nan  # 如果所有真实值都接近零

    relative_errors = np.abs(y_true_arr[valid_indices] - y_pred_arr[valid_indices]) / (
        np.abs(y_true_arr[valid_indices]))
    return np.mean(relative_errors) * 100


def calculate_r2(y_true, y_pred):
    """计算决定系数 R2"""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    if ss_tot < 1e-9:  # 如果所有y_true几乎相同
        return 1.0 if ss_res < 1e-9 else 0.0  # 如果残差也很小，则R2为1，否则为0
    return 1 - (ss_res / ss_tot)


# ------------------------------------------------------------------------
# 模型1: Boltzmann 时间函数拟合
# ------------------------------------------------------------------------
print("\n--- Boltzmann 模型拟合 ---")


def Boltzmann(t_data, A_param, t0_param, B_param):
    # A: 振幅 (最大下沉)
    # t0: 拐点时间 (约一半最大下沉时的时间)
    # B: 曲线陡峭程度相关参数
    exponent = (t_data - t0_param) / (B_param + 1e-9)  # 加epsilon防止B_param为0
    exponent_clipped = np.clip(exponent, -700, 700)  # 限制指数范围防止溢出
    # 模型形式调整为输出正值，因为 wt = abs(wt)
    return A_param - A_param / (1 + np.exp(exponent_clipped))

# 设置初始参数和边界
initial_guess_boltz = [A_observed_max, t_observed_median, t_observed_median / 4 if t_observed_median > 1e-6 else 1.0]
bounds_boltz = (
    [0.1 * A_observed_max, 0, 1e-6],  # 下界 [A, t0, B] (B > 0)
    [2.0 * A_observed_max, t_observed_max * 2, t_observed_max * 2]  # 上界
)

try:
    params_boltz, pcov_boltz = curve_fit(Boltzmann, t, wt, p0=initial_guess_boltz, bounds=bounds_boltz, maxfev=5000)
    A, t0, B = params_boltz
    wt_pred_boltz = Boltzmann(t, A, t0, B)

    print(f"拟合参数: A = {A:.4f}, t0 = {t0:.4f}, B = {B:.4f}")
    r2_boltz = calculate_r2(wt, wt_pred_boltz)
    rmse_boltz = calculate_rmse(wt, wt_pred_boltz)
    relative_error_boltz = calculate_relative_error(wt, wt_pred_boltz)
    print(f"R²: {r2_boltz:.4f}")
    print(f"RMSE: {rmse_boltz:.4f}")
    print(f"相对误差: {relative_error_boltz:.4f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(t, wt, c='r', label='实测数据', marker='o')
    plt.plot(t, wt_pred_boltz, c='b', label=f'Boltzmann拟合\nA={A:.2f}, t0={t0:.2f}, B={B:.2f}\nR²={r2_boltz:.3f}')
    plt.xlabel("时间间隔 (t)")
    plt.ylabel("下沉值 (wt)")
    plt.title("Boltzmann 时间函数拟合")
    plt.legend()
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"Boltzmann拟合失败: {e}")
except OptimizeWarning as ow:  # curve_fit 可以抛出 OptimizeWarning
    print(f"Boltzmann拟合警告: {ow}")
    if _ is None:  # _ 是 pcov，如果为None，通常表示协方差未估计
        print("参数的协方差矩阵未能成功估计。")

# ------------------------------------------------------------------------
# 模型2: Weibull 时间函数模型拟合
# 原始模型: -2707*(1-np.exp(-c*(t**k)))
# 我们将幅值参数化，并确保输出为正
# ------------------------------------------------------------------------
print("\n--- Weibull 模型拟合 ---")
# W_max_weibull = 2707 # 您原始的硬编码值
W_max_weibull = A_observed_max  # 或者使用观测到的最大值


def Weibull(t_data, c_param, k_param, Wmax_param):  # Wmax作为参数
    exponent = -c_param * (np.power(t_data, k_param) + 1e-9)  # 加epsilon防止t_data为0且k<0
    exponent_clipped = np.clip(exponent, -700, 700)
    return Wmax_param * (1 - np.exp(exponent_clipped))


initial_guess_weibull = [0.1, 1.0, W_max_weibull]  # c, k, Wmax
bounds_weibull = (
    [1e-6, 1e-6, 0.1 * W_max_weibull],  # 下界 [c, k, Wmax]
    [100, 10, 2.0 * W_max_weibull]  # 上界
)

try:
    params_weibull, pcov_weibull = curve_fit(Weibull, t, wt, p0=initial_guess_weibull, bounds=bounds_weibull,
                                             maxfev=5000)
    c, k, Wmax_fit = params_weibull  # 获取拟合后的Wmax
    wt_pred_weibull = Weibull(t, c, k, Wmax_fit)

    print(f"拟合参数: Wmax = {Wmax_fit:.4f}, c = {c:.4f}, k = {k:.4f}")
    r2_weibull = calculate_r2(wt, wt_pred_weibull)
    rmse_weibull = calculate_rmse(wt, wt_pred_weibull)
    relative_error_weibull = calculate_relative_error(wt, wt_pred_weibull)
    print(f"R²: {r2_weibull:.4f}")
    print(f"RMSE: {rmse_weibull:.4f}")
    print(f"相对误差: {relative_error_weibull:.4f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(t, wt, c='r', label='实测数据', marker='o')
    plt.plot(t, wt_pred_weibull, c='b',
             label=f'Weibull拟合\nWmax={Wmax_fit:.2f}, c={c:.3f}, k={k:.3f}\nR²={r2_weibull:.3f}')
    plt.xlabel("时间间隔 (t)")
    plt.ylabel("下沉值 (wt)")
    plt.title("Weibull 时间函数拟合")
    plt.legend()
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"Weibull拟合失败: {e}")
except OptimizeWarning as ow:
    print(f"Weibull拟合警告: {ow}")
    if pcov_weibull is None:
        print("参数的协方差矩阵未能成功估计。")

# ------------------------------------------------------------------------
# 模型3: Logistic 时间函数模型拟合
# 原始模型: -2707*(1-1/(1+(t/x0)**p))
# ------------------------------------------------------------------------
print("\n--- Logistic 模型拟合 ---")
# W_max_logistic = 2707
W_max_logistic = A_observed_max


def Logistic(t_data, x0_param, p_param, Wmax_param):  # Wmax作为参数
    # 确保 x0_param > 0
    ratio = t_data / (x0_param + 1e-9)  # 防止 x0_param 为0
    # 如果 p_param 可能为非整数，则 ratio 必须为非负
    # 或者确保 (t/x0) 不会因为 p 次方而产生复数或错误
    base_clipped = np.clip(ratio, 0, 1e6)  # 限制底数防止过大或负数问题（如果p非整数）
    term = np.power(base_clipped, p_param)
    return Wmax_param * (term / (1 + term))  # (1 - 1/(1+X)) = X/(1+X)


initial_guess_logistic = [t_observed_median, 2.0, W_max_logistic]  # x0, p, Wmax
bounds_logistic = (
    [1e-6, 1e-6, 0.1 * W_max_logistic],  # 下界 [x0, p, Wmax]
    [t_observed_max * 2, 20, 2.0 * W_max_logistic]  # 上界
)

try:
    params_logistic, pcov_logistic = curve_fit(Logistic, t, wt, p0=initial_guess_logistic, bounds=bounds_logistic,
                                               maxfev=5000)
    x0, p, Wmax_fit_log = params_logistic
    wt_pred_logistic = Logistic(t, x0, p, Wmax_fit_log)

    print(f"拟合参数: Wmax = {Wmax_fit_log:.4f}, x0 = {x0:.4f}, p = {p:.4f}")
    r2_logistic = calculate_r2(wt, wt_pred_logistic)
    rmse_logistic = calculate_rmse(wt, wt_pred_logistic)
    relative_error_logistic = calculate_relative_error(wt, wt_pred_logistic)
    print(f"R²: {r2_logistic:.4f}")
    print(f"RMSE: {rmse_logistic:.4f}")
    print(f"相对误差: {relative_error_logistic:.4f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(t, wt, c='r', label='实测数据', marker='o')
    plt.plot(t, wt_pred_logistic, c='b',
             label=f'Logistic拟合\nWmax={Wmax_fit_log:.2f}, x0={x0:.2f}, p={p:.2f}\nR²={r2_logistic:.3f}')
    plt.xlabel("时间间隔 (t)")
    plt.ylabel("下沉值 (wt)")
    plt.title("Logistic 时间函数拟合")
    plt.legend()
    plt.grid(True)
    plt.show()
except RuntimeError as e:
    print(f"Logistic拟合失败: {e}")
except OptimizeWarning as ow:
    print(f"Logistic拟合警告: {ow}")
    if pcov_logistic is None:
        print("参数的协方差矩阵未能成功估计。")

# ------------------------------------------------------------------------
# 模型4: Bertalanffy 时间函数模型拟合
# 原始模型: -2707*((1-b*np.exp(-c*t))**d)
# ------------------------------------------------------------------------
print("\n--- Bertalanffy 模型拟合 ---")
# W_max_bert = 2707
W_max_bert = A_observed_max


def Bertalanffy(t_data, b_param, c_param, d_param, Wmax_param):  # Wmax作为参数
    exponent = -c_param * t_data
    exponent_clipped = np.clip(exponent, -700, 700)
    base = (1 - b_param * np.exp(exponent_clipped))
    # 如果 d_param 可能为非整数，则 base 必须为非负
    base_clipped = np.maximum(base, 1e-9)  # 防止base为0或负数，且d_param非整数
    return Wmax_param * (np.power(base_clipped, d_param))


initial_guess_bert = [0.5, 0.01, 1.0, W_max_bert]  # b, c, d, Wmax
bounds_bert = (
    [1e-6, 1e-6, 1e-6, 0.1 * W_max_bert],  # 下界 [b, c, d, Wmax]
    [1 - 1e-6, 10, 10, 2.0 * W_max_bert]  # 上界 (b < 1)
)

try:
    params_bert, pcov_bert = curve_fit(Bertalanffy, t, wt, p0=initial_guess_bert, bounds=bounds_bert, maxfev=5000)
    b, c, d, Wmax_fit_bert = params_bert
    wt_pred_bert = Bertalanffy(t, b, c, d, Wmax_fit_bert)

    print(f"拟合参数: Wmax = {Wmax_fit_bert:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}")
    r2_bert = calculate_r2(wt, wt_pred_bert)
    rmse_bert = calculate_rmse(wt, wt_pred_bert)
    relative_error_bert = calculate_relative_error(wt, wt_pred_bert)
    print(f"R²: {r2_bert:.4f}")
    print(f"RMSE: {rmse_bert:.4f}")
    print(f"相对误差: {relative_error_bert:.4f}%")

    # y_predict=Bertalanffy(t,b,c,d) # 旧的，应该用拟合后的Wmax
    plt.figure(figsize=(10, 6))
    plt.scatter(t, wt, c='r', label='实测数据', marker='o')
    plt.plot(t, wt_pred_bert, c='b',
             label=f'Bertalanffy拟合\nWmax={Wmax_fit_bert:.2f}, b={b:.3f}, c={c:.3f}, d={d:.3f}\nR²={r2_bert:.3f}')
    plt.xlabel("时间间隔 (t)")
    plt.ylabel("下沉值 (wt)")
    plt.title("Bertalanffy 时间函数拟合")
    plt.legend()
    plt.grid(True)
    plt.show()
except RuntimeError as e:
    print(f"Bertalanffy拟合失败: {e}")
except OptimizeWarning as ow:
    print(f"Bertalanffy拟合警告: {ow}")
    if pcov_bert is None:
        print("参数的协方差矩阵未能成功估计。")