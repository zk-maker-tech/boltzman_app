import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 读取excel文件
file_path = 'coal_data.xlsx'
dataframe = pd.read_excel(file_path,sheet_name=0)
t = dataframe.iloc[:, 3]  # 时间间隔列

# Boltzmann时间函数拟合
def Boltzmann(t, A, t0, B):
    exponent = np.clip((t - t0) / B, -700, 700)
    return -A / (1 + np.exp(exponent)) + A

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

results = []

for col_index in range(4, dataframe.shape[1]):
    col_name = dataframe.columns[col_index]
    wt_original = dataframe.iloc[:, col_index]
    wt = np.abs(wt_original)

    # 鲁棒性检查
    if not np.any(wt > 0):
        print(f"跳过列 {col_name}：所有值都为零或负数")
        continue
    if np.any(np.isnan(wt)) or np.any(np.isinf(wt)):
        print(f"跳过列 {col_name}：包含 NaN 或无穷大")
        continue

    # 初始参数猜测
    initial_guess = [np.max(wt), np.median(t), np.std(t)]

    try:
        params, _ = curve_fit(Boltzmann, t, wt, p0=initial_guess)
        A, t0, B = params
        r2 = calculate_r2(wt, Boltzmann(t, A, t0, B))
        Wm = np.max(wt)
        relative_error = (A - Wm) / Wm if Wm != 0 else np.nan

        results.append({
            "列名": col_name,
            "A": A,
            "t0": t0,
            "B": B,
            "R²": r2,
            "Wm": Wm,
            "相对误差": relative_error
        })
    except RuntimeError:
        print(f"拟合失败：列 {col_name}")

# 打印所有结果
for res in results:
    print(f" {res['列名']}, A={res['A']:.3f}, t0={res['t0']:.3f}, B={res['B']:.3f}, R²={res['R²']:.3f}, Wm={res['Wm']:.3f}, 相对误差={res['相对误差']:.3%}")


