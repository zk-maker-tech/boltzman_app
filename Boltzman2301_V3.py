import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 读取excel文件
file_path = 'coal_data.xlsx'
dataframe = pd.read_excel(file_path,sheet_name=0)
t = dataframe.iloc[:, 3]  # 时间间隔列

# Boltzmann时间函数拟合
def Boltzmann(t_series, A, t0, B): # 将参数 t 重命名为 t_series 避免与外部变量混淆
    exponent = np.clip((t_series - t0) / B, -700, 700)
    return -A / (1 + np.exp(exponent)) + A

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: # 处理 y_true 全为常数的情况，避免除以零
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

results = []
# 新增一个字典来存储每列的拟合点值
fitted_values_all_columns = {}

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
    # 确保初始猜测的A是正值，因为Boltzmann函数通常A>0
    initial_A_guess = np.max(wt) if np.max(wt) > 0 else 1.0
    initial_t0_guess = np.median(t) if not t.empty else 0 # 确保t不为空
    initial_B_guess = np.std(t) if not t.empty and np.std(t) != 0 else 1.0 # 确保B不为0

    initial_guess = [initial_A_guess, initial_t0_guess, initial_B_guess]


    try:
        params, _ = curve_fit(Boltzmann, t, wt, p0=initial_guess, maxfev=5000) # 增加maxfev迭代次数
        A_fit, t0_fit, B_fit = params # 使用不同的变量名存储拟合参数

        # --- 提取拟合曲线对应点的值 ---
        wt_fitted = Boltzmann(t, A_fit, t0_fit, B_fit)
        # 将当前列的拟合点值存储起来
        fitted_values_all_columns[col_name] = wt_fitted
        # ---------------------------------

        r2 = calculate_r2(wt, wt_fitted) # 使用已计算的 wt_fitted
        Wm = np.max(wt)
        relative_error = (A_fit - Wm) / Wm if Wm != 0 else np.nan

        results.append({
            "列名": col_name,
            "A": A_fit,
            "t0": t0_fit,
            "B": B_fit,
            "R²": r2,
            "Wm": Wm,
            "相对误差": relative_error
            # 如果需要，也可以把拟合点序列加入到results字典中
            # "拟合值序列": wt_fitted.tolist() # 转换为列表形式
        })
    except RuntimeError:
        print(f"拟合失败：列 {col_name}")
    except ValueError as e:
        print(f"拟合时发生数值错误：列 {col_name} - {e}")


# 打印所有结果
print("\n--- 拟合参数结果 ---")
for res in results:
    print(f" {res['列名']}, A={res['A']:.3f}, t0={res['t0']:.3f}, B={res['B']:.3f}, R²={res['R²']:.3f}, Wm={res['Wm']:.3f}, 相对误差={res['相对误差']:.3%}")

# 打印提取的拟合点值
print("\n--- 拟合曲线上对应点的值 ---")
for col_name, fitted_vals in fitted_values_all_columns.items():
    print(f"\n列: {col_name}")
    # print(f"时间点 (t):\n{t.values}") # 原始时间点
    # print(f"原始观测值 (wt):\n{dataframe[col_name].abs().values}") # 原始观测值 (取绝对值后)
    # print(f"拟合曲线上的值 (wt_fitted):\n{fitted_vals}")

    # 更清晰地并排显示 (如果数据点不多的话)
    print("时间 (t) | 原始值 (wt) | 拟合值 (wt_fitted)")
    print("---------------------------------------------")
    # 确保dataframe[col_name].abs().values 和 t 以及 fitted_vals 长度一致
    # 如果之前处理过NaN导致t和wt长度不一致，这里需要使用对应拟合时所用的t
    # 但在这个版本的代码中，t是全局的，如果t中有NaN，curve_fit可能在之前就失败了
    # 假设t是干净的，并且长度与wt_original一致
    original_wt_for_column = dataframe.iloc[:, dataframe.columns.get_loc(col_name)].abs().values
    for t_val, original_val, fitted_val in zip(t, original_wt_for_column, fitted_vals):
        print(f"{t_val:8.2f} | {original_val:10.3f} | {fitted_val:15.3f}")