import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 读取excel文件
file_path = 'coal_data.xlsx'
dataframe = pd.read_excel(file_path,sheet_name=0)
t = dataframe.iloc[:, 1]  # 时间间隔

########################################################################
# Boltzmann时间函数
def Boltzmann(t, A, t0, B):
    exponent = np.clip((t - t0) / B, -700, 700)
    return -A / (1 + np.exp(exponent)) + A

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: # 处理y_true全为常数的情况
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

# 存储所有拟合结果的列表
results_list = []

# 从第4列开始到最后一列进行拟合
for i in range(4, dataframe.shape[1]):
    wt_original = dataframe.iloc[:, i]  # 下沉值
    column_name = dataframe.columns[i]
    print(f"\n正在处理列: {column_name}")

    wt = np.abs(wt_original)  # 转换为正值

    # 移除NaN值 (如果时间t和wt长度不一致，curve_fit会报错)
    valid_indices = ~np.isnan(wt)
    t_filtered = t[valid_indices].to_numpy() # Convert to numpy array for curve_fit
    wt_filtered = wt[valid_indices].to_numpy() # Convert to numpy array

    if len(t_filtered) == 0 or len(wt_filtered) == 0:
        print(f"列 {column_name} 不包含有效数据，跳过。")
        results_list.append({
            '列名': column_name,
            'A': np.nan,
            't0': np.nan,
            'B': np.nan,
            'R2': np.nan,
            'Wm_实测': np.nan,
            '相对误差': np.nan,
            '备注': '无有效数据'
        })
        continue

    # 鲁棒性检查
    if not np.any(wt_filtered > 0):
        print(f"错误：列 {column_name} 的 wt 数据中无正值，可能需要检查数据或模型。")
        results_list.append({
            '列名': column_name,
            'A': np.nan,
            't0': np.nan,
            'B': np.nan,
            'R2': np.nan,
            'Wm_实测': np.nan,
            '相对误差': np.nan,
            '备注': 'wt无正值'
        })
        continue
    # if np.any(np.isnan(wt_filtered)) or np.any(np.isinf(wt_filtered)): # Already handled by dropna and valid_indices
    #     print(f"错误：列 {column_name} 的 wt 包含 NaN 或无穷大值（清理后仍存在）。")
    #     results_list.append({
    #         '列名': column_name,
    #         'A': np.nan, 't0': np.nan, 'B': np.nan, 'R2': np.nan,
    #         'Wm_实测': np.nan, '相对误差': np.nan, '备注': 'wt包含NaN或无穷大值'
    #     })
    #     continue
    if len(t_filtered) < 3: # curve_fit至少需要与参数数量相同的数据点
        print(f"错误：列 {column_name} 的有效数据点不足 ({len(t_filtered)}) 进行拟合。")
        results_list.append({
            '列名': column_name,
            'A': np.nan,
            't0': np.nan,
            'B': np.nan,
            'R2': np.nan,
            'Wm_实测': np.max(wt_filtered) if len(wt_filtered) > 0 else np.nan,
            '相对误差': np.nan,
            '备注': '数据点不足'
        })
        continue

    # 初始参数猜测
    initial_A_guess = np.max(wt_filtered) if len(wt_filtered) > 0 else 1.0
    initial_t0_guess = np.median(t_filtered) if len(t_filtered) > 0 else np.mean(t.dropna()) # Use original t if t_filtered is empty
    initial_B_guess = np.std(t_filtered) if len(t_filtered) > 1 else 1.0

    initial_A_guess = initial_A_guess if np.isfinite(initial_A_guess) and initial_A_guess > 0 else 1.0
    initial_t0_guess = initial_t0_guess if np.isfinite(initial_t0_guess) else (np.mean(t.dropna()) if not t.dropna().empty else 0)
    initial_B_guess = initial_B_guess if np.isfinite(initial_B_guess) and initial_B_guess != 0 else 1.0

    initial_guess = [initial_A_guess, initial_t0_guess, initial_B_guess]

    try:
        # 使用 curve_fit 进行拟合
        # Ensure t_filtered and wt_filtered are 1D numpy arrays of the same finite size
        if t_filtered.ndim != 1 or wt_filtered.ndim != 1 or t_filtered.size != wt_filtered.size or not np.all(np.isfinite(t_filtered)) or not np.all(np.isfinite(wt_filtered)):
            raise ValueError("Input arrays for curve_fit are not suitable.")

        params, _ = curve_fit(Boltzmann, t_filtered, wt_filtered, p0=initial_guess, maxfev=5000)

        # 获取拟合后的参数
        A_fit, t0_fit, B_fit = params

        print(f"列 {column_name} 拟合得到的参数: A = {A_fit}, t0 = {t0_fit}, B = {B_fit}")
        r2 = calculate_r2(wt_filtered, Boltzmann(t_filtered, A_fit, t0_fit, B_fit))
        print(f"列 {column_name} 相关系数: r2 = {r2}")

        Wm = np.max(wt_filtered) if len(wt_filtered) > 0 else np.nan # 使用最大下沉值
        print(f"列 {column_name} 实测最大下沉值: Wm = {Wm}")
        relative_error = (A_fit - Wm) / Wm if Wm != 0 and not np.isnan(Wm) else np.nan  # 防止除零
        print(f"列 {column_name} 相对误差: relative_error = {relative_error}")

        results_list.append({
            '列名': column_name,
            'A': A_fit,
            't0': t0_fit,
            'B': B_fit,
            'R2': r2,
            'Wm_实测': Wm,
            '相对误差': relative_error,
            '备注': '拟合成功'
        })

        # # 绘图代码段已被移除
        # plt.figure()
        # plt.plot(t_filtered, wt_filtered, 'o', c='r', label='实测数据')
        # t_plot = np.linspace(min(t_filtered), max(t_filtered), 200) # Requires t_filtered to be non-empty
        # plt.plot(t_plot, Boltzmann(t_plot, A_fit, t0_fit, B_fit), c='b', label=f'Boltzmann拟合\nA={A_fit:.2f}, t0={t0_fit:.2f}, B={B_fit:.2f}\nR²={r2:.3f}')
        # plt.xlabel("时间间隔 (t/d)")
        # plt.ylabel("下沉值 (wt)")
        # plt.title(f" Boltzmann 时间函数拟合 - {column_name}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    except RuntimeError:
        print(f"列 {column_name} 无法进行拟合。可能是初始猜测不佳或数据不适合此模型。")
        results_list.append({
            '列名': column_name,
            'A': np.nan,
            't0': np.nan,
            'B': np.nan,
            'R2': np.nan,
            'Wm_实测': np.max(wt_filtered) if len(wt_filtered) > 0 else np.nan,
            '相对误差': np.nan,
            '备注': '拟合失败 (RuntimeError)'
        })
    except ValueError as e:
        print(f"列 {column_name} 拟合时发生数值错误: {e}")
        results_list.append({
            '列名': column_name,
            'A': np.nan,
            't0': np.nan,
            'B': np.nan,
            'R2': np.nan,
            'Wm_实测': np.max(wt_filtered) if len(wt_filtered) > 0 else np.nan,
            '相对误差': np.nan,
            '备注': f'拟合失败 (ValueError: {e})'
        })


# 将结果汇总到DataFrame
results_df = pd.DataFrame(results_list)
print("\n\n--- 拟合结果汇总 ---")
print(results_df)

# 可以选择将结果保存到新的Excel文件
# results_df.to_excel("boltzmann_fitting_results_no_plots.xlsx", index=False)