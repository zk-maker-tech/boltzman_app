import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# === 预设 A 值字典 ===
preset_A_map = dict(zip(
    [
        'RQ01', 'RQ02', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08',
        'Q09', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018',
        'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025', 'Q026', 'Q027', 'Q028',
        'Q029', 'Q035', 'Q036', 'RQ03', 'RQ04'
    ],
    [
        23, 18, 15, 21, 39, 56, 73, 11, 52, 66,
        63, 102, 232, 348, 540, 879, 1023, 1139, 871, 666,
        373, 198, 204, 84, 57, 67, 46, 41, 42, 32,
        39, -4, -6, -6, 4
    ]
))

# === Excel 数据读取 ===
file_path = 'coal_data.xlsx'
try:
    df = pd.read_excel(file_path,sheet_name='up')
except FileNotFoundError:
    print(f"文件未找到：{file_path}")
    exit()
except Exception as e:
    print(f"读取失败：{e}")
    exit()

t_fit_series = df.iloc[:, 1]
t_pred_series = df.iloc[:, 3]
t_pred = np.asarray(t_pred_series)

# === Boltzmann 模型 ===
def Boltzmann(t, A, t0, B):
    exp_val = np.clip((t - t0) / B, -700, 700)
    return -A / (1 + np.exp(exp_val)) + A

# === R² 计算 ===
def r2_score(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0: return np.nan
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res < 1e-9 else 0.0)

# === 主处理逻辑 ===
fit_results, curves_out = [], pd.DataFrame({'Prediction_Time': t_pred})

for i in range(4, df.shape[1]):
    col = df.columns[i]
    y = np.abs(df.iloc[:, i])
    t = t_fit_series
    data = pd.DataFrame({'t': t, 'y': y}).dropna()
    t_fit, y_fit = np.asarray(data['t']), np.asarray(data['y'])

    remark = ''
    if len(t_fit) < 3 or not np.any(y_fit > 0):
        remark = f'数据点不足或无正值: {len(t_fit)}'
    elif not np.all(np.isfinite(t_fit)) or not np.all(np.isfinite(y_fit)):
        remark = '包含非有限值'

    if remark:
        print(f"跳过 {col}: {remark}")
        fit_results.append({'列名': col, '备注': remark, 'Wm_实测_拟合用': np.max(y_fit) if len(y_fit) else np.nan})
        continue

    # 初始参数
    A0, t0_0, B0 = np.max(y_fit), np.median(t_fit), np.std(t_fit) or 1e-3
    try:
        params, _ = curve_fit(Boltzmann, t_fit, y_fit, p0=[A0, t0_0, B0], maxfev=8000)
        A_fit, t0, B = params
        A_used = preset_A_map.get(col, A_fit)
        remark = f"使用{'预设' if col in preset_A_map else '拟合'}A: {A_used:.3f}"
        y_pred = Boltzmann(t_pred, A_used, t0, B)
        curves_out[col] = y_pred
        r2_val = r2_score(y_fit, Boltzmann(t_fit, A_used, t0, B))
        rel_err = (A_used - np.max(y_fit)) / np.max(y_fit) if np.max(y_fit) else np.nan
        print(f"{col} 拟合完成: A={A_fit:.2f}, A_used={A_used:.2f}, t0={t0:.2f}, B={B:.2f}")
    except Exception as e:
        remark = f"拟合失败: {type(e).__name__}: {e}"
        A_used = A_fit = t0 = B = r2_val = rel_err = np.nan

    fit_results.append({
        '列名': col, 'A_used': A_used, 'A_fitted': A_fit, 't0': t0, 'B': B,
        'R2': r2_val, 'Wm_实测_拟合用': np.max(y_fit), '相对误差': rel_err, '备注': remark
    })

# === 拟合汇总输出 ===
summary_df = pd.DataFrame(fit_results)
ordered_cols = ['列名', 'A_used', 'A_fitted', 't0', 'B', 'R2', 'Wm_实测_拟合用', '相对误差', '备注']
print("\n--- 拟合总结 ---")
print(summary_df[ordered_cols].to_string(index=False, float_format='%.3f'))

# === 曲线结果保存 ===
if curves_out.shape[1] > 1:
    try:
        out_path = "boltzmann_bottom.xlsx"
        curves_out.to_excel(out_path, index=False, float_format="%.3f")
        print(f"\n✅ 曲线已保存: {out_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
else:
    print("⚠️ 没有生成任何拟合曲线")

print("\n🎉 脚本执行完毕。")
