import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D
from custom_legend_handler_v2 import HandlerInterspersedText

# === 配置区 ===
file_path_curves = 'boltzmann.xlsx'
file_path_markers = 'coal_data.xlsx'
prediction_time_key_options = ['Prediction_time', 'Prediction_Time']
plot_filename_q = 'q_series_plot_no_legend_frame.png'

# === 实际空间坐标（与 Q 列对应） ===
x_coordinates_q = np.array([
    -402.231, -352.89, -301.064, -274.51, -253.133,
    -227.327, -203.713, -175.675, -148.62, -127.857,
    -101.91, -78.263, -39.105, -20.13, 4.093,
    34.095, 45.119, 73.094, 97.966, 122.818,
    147.73, 168.246, 190.122, 230.184, 243.724,
    288.408, 298.511, 323.341, 346.781, 370.381,
    397.568, 495.621, 525.896, 572.645, 624.182
])

# === 图表字体设置 ===
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# === 工具函数 ===
def read_data(file_path, fallback_csv=None):
    try:
        return pd.read_excel(file_path)
    except Exception:
        if fallback_csv:
            try:
                return pd.read_csv(fallback_csv)
            except Exception as e:
                raise FileNotFoundError(f"读取失败: {file_path}, fallback: {e}")
        else:
            raise FileNotFoundError(f"读取失败: {file_path}")

def find_key_column(df, options):
    return next((col for col in options if col in df.columns), None)

def sanitize_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"跳过列：{missing}（在数据中不存在）")
    columns = [col for col in columns if col in df.columns]
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=columns).copy(), columns

def safe_spline(x, y):
    try:
        return make_interp_spline(x, y, k=min(3, len(x)-1))
    except Exception:
        return None

def create_legend_entry(color, has_actual, time_label):
    line = Line2D([0], [0], color=color, linestyle='-')
    if has_actual:
        marker = Line2D([0], [0], marker='.', color=color, linestyle='None', markersize=8)
        handler = HandlerInterspersedText("预计", f"实际 ({time_label})")
        return (line, marker), handler
    return line, None

def plot_legend(ax, handles, labels, handler_map):
    loc = 'best' if len(handles) <= 7 else 'upper left'
    box = None if loc == 'best' else (1.05, 1)
    ax.legend(handles, labels, title="图例", loc=loc, bbox_to_anchor=box,
              fontsize=10, handlelength=10.0, numpoints=1, markerscale=0.8,
              labelspacing=0.8, frameon=False, handler_map=handler_map)
    plt.tight_layout(rect=[0, 0, 0.85, 1] if box else None)

# === 主流程 ===
try:
    df_curves = read_data(file_path_curves, fallback_csv=file_path_curves.replace(".xlsx", " - Sheet1.csv"))
    df_markers = read_data(file_path_markers, fallback_csv='coal_data.xlsx - up.csv')

    key_curve = find_key_column(df_curves, prediction_time_key_options)
    key_marker = find_key_column(df_markers, prediction_time_key_options)

    if not key_curve or not key_marker:
        raise ValueError("未找到预测时间关键列")

    raw_q_columns = df_markers.columns[4:].tolist()
    df_curves, q_columns = sanitize_columns(df_curves, raw_q_columns)
    df_curves[key_curve] = df_curves[key_curve].astype(str)

    x_coordinates_q = x_coordinates_q[:len(q_columns)]
    if len(q_columns) != len(x_coordinates_q):
        raise ValueError(f"q_columns 与 x_coordinates_q 长度不一致: {len(q_columns)} vs {len(x_coordinates_q)}")

    df_markers_subset = df_markers[[key_marker] + q_columns].copy()
    df_markers_subset[key_marker] = df_markers_subset[key_marker].astype(str)
    df_markers_renamed = df_markers_subset.rename(columns={col: f"{col}_marker" for col in q_columns})

    merged = pd.merge(df_curves, df_markers_renamed, left_on=key_curve, right_on=key_marker, how='left')

    fig, ax = plt.subplots(figsize=(14, 8))
    legend_handles, legend_labels, handler_map = [], [], {}

    for _, row in merged.iterrows():
        y_curve = -np.abs(row[q_columns].values.astype(float))
        spline = safe_spline(x_coordinates_q, y_curve)

        if spline:
            x_smooth = np.linspace(x_coordinates_q.min(), x_coordinates_q.max(), 300)
            y_smooth = spline(x_smooth)
            line, = ax.plot(x_smooth, y_smooth)
        else:
            line, = ax.plot(x_coordinates_q, y_curve)

        color = line.get_color()
        marker_cols = [f"{col}_marker" for col in q_columns if f"{col}_marker" in row.index]
        y_marker = -np.abs(row[marker_cols].values.astype(float))
        valid = ~np.isnan(y_marker)
        has_actual = valid.any()

        if has_actual:
            ax.plot(x_coordinates_q[valid], y_marker[valid], marker='.', linestyle='None', color=color, markersize=8)

        time_label = row[key_curve]
        legend_entry, handler = create_legend_entry(color, has_actual, time_label)
        legend_handles.append(legend_entry)
        legend_labels.append("") if handler else legend_labels.append(f"预计 ({time_label}t)")
        if handler:
            handler_map[legend_entry] = handler

    ax.set_xlabel("指定空间坐标 (Index Coordinate for Q-series)")
    ax.set_ylabel("Q系列数值 (下沉 - 负值)")
    ax.set_title("Q系列下沉数据")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ticks = np.linspace(x_coordinates_q.min(), x_coordinates_q.max(), 5).round(1)
    ax.set_xticks(ticks)

    plot_legend(ax, legend_handles, legend_labels, handler_map)
    plt.savefig(plot_filename_q)
    print(f"图表已保存为 {plot_filename_q}")

except Exception as e:
    import traceback
    print(f"发生错误: {e}")
    traceback.print_exc()

print("\n--- Q系列绘图脚本结束 ---")
