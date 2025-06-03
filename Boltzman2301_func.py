import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# === 读取 Excel 数据 ===
def read_excel_data(filepath, sheet_index=0):
    try:
        return pd.read_excel(filepath, sheet_name=sheet_index)
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
    return None

# === Boltzmann 函数 ===
def boltzmann(t, A, t0, B):
    exponent = np.clip((t - t0) / B, -700, 700)  # 避免 exp 溢出
    return -A / (1 + np.exp(exponent)) + A

# === R² 计算函数 ===
def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else (1.0 if ss_res < 1e-9 else 0.0)

# === 准备拟合数据 ===
def prepare_fit_data(t_series, wt_series):
    df = pd.DataFrame({'t': t_series, 'wt': np.abs(wt_series)}).dropna()
    return df['t'].to_numpy(), df['wt'].to_numpy()

# === 主函数 ===
def main():
    filepath = 'coal_data.xlsx'
    df = read_excel_data(filepath)
    if df is None:
        return

    t_fitting = df.iloc[:, 1]
    time_pred = df.iloc[:, 3].to_numpy()
    result_df = pd.DataFrame({'Prediction_Time': time_pred})
    summary = []

    for idx in range(4, df.shape[1]):
        col_name = df.columns[idx]
        print(f"\n处理列: {col_name}")
        wt_series = df.iloc[:, idx]
        t_fit, wt_fit = prepare_fit_data(t_fitting, wt_series)

        if len(t_fit) < 3 or not np.any(wt_fit > 0):
            reason = '数据点不足' if len(t_fit) < 3 else 'Wt无正值'
            print(f"  跳过 {col_name}: {reason}")
            summary.append({
                '列名': col_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
                'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit) if len(wt_fit) > 0 else np.nan,
                '相对误差': np.nan, '备注': reason
            })
            continue

        A0, t0_0, B0 = np.max(wt_fit), np.median(t_fit), np.std(t_fit) or 1e-3
        init_guess = [A0, t0_0, B0]

        if pd.isnull(init_guess).any():
            print(f"  跳过 {col_name}: 初始猜测含NaN")
            summary.append({
                '列名': col_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
                'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit), '相对误差': np.nan, '备注': '初始猜测含NaN'
            })
            continue

        try:
            params, _ = curve_fit(boltzmann, t_fit, wt_fit, p0=init_guess, maxfev=8000)
            A, t0, B = params
            pred_curve = boltzmann(time_pred, A, t0, B)
            result_df[col_name] = pred_curve

            r2 = r2_score(wt_fit, boltzmann(t_fit, A, t0, B))
            Wm_actual = np.max(wt_fit)
            rel_err = (A - Wm_actual) / Wm_actual if Wm_actual else np.nan

            print(f"  拟合成功: A={A:.3f}, t0={t0:.3f}, B={B:.3f}")
            summary.append({
                '列名': col_name, 'A': A, 't0': t0, 'B': B,
                'R2': r2, 'Wm_实测_拟合用': Wm_actual,
                '相对误差': rel_err, '备注': '拟合成功'
            })

        except Exception as e:
            print(f"  拟合失败: {e}")
            summary.append({
                '列名': col_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
                'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit) if len(wt_fit) > 0 else np.nan,
                '相对误差': np.nan, '备注': f'拟合失败 ({type(e).__name__})'
            })

    # 输出结果
    summary_df = pd.DataFrame(summary)
    print("\n--- 拟合摘要 ---")
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(summary_df if not summary_df.empty else "无拟合数据。")

    print("\n--- Boltzmann预测曲线（前5行） ---")
    print(result_df.head() if result_df.shape[1] > 1 else "无预测结果。")

    # 写入 Excel
    try:
        output_file = "calculated_boltzmann_curves_with_col4_time.xlsx"
        result_df.to_excel(output_file, index=False, float_format="%.3f")
        print(f"\n预测结果已保存至: {output_file}")
    except Exception as e:
        print(f"\n保存Excel时出错: {e}")

if __name__ == "__main__":
    main()
