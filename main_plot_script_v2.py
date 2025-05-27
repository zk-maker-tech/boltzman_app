# main_plot_script.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

# 从我们自定义的模块中导入 HandlerInterspersedText 类
# 确保 custom_legend_handler.py 与此脚本在同一目录
from custom_legend_handler_v2 import HandlerInterspersedText

# --- 全局字体设置 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置信息 ---
file_path_curves = 'fit.xlsx'
file_path_markers = 'coal_data.xlsx'
q_columns = [f'Q{i:03}' for i in range(11, 21)]
x_coordinates_q = np.array([
    -39.105, -20.13, 4.093, 34.095, 45.119, 73.094,
    97.966, 122.818, 147.73, 168.246
])
prediction_time_key_options = ['Prediction_time', 'Prediction_Time']
plot_filename_q = 'q_series_plot_no_legend_frame.png'  # 更新文件名


def get_key_column(df, options):
    for col_name in options:
        if col_name in df.columns:
            return col_name
    return None


# --- 主程序逻辑 ---
try:
    # 1. 加载数据
    try:
        df_curves = pd.read_excel(file_path_curves)
    except Exception as e_curves_excel:
        print(f"提示: 读取 '{file_path_curves}' Excel 文件失败 ({e_curves_excel}). 尝试读取 CSV.")
        try:
            csv_curve_path = file_path_curves.replace(".xlsx",
                                                      " - Sheet1.csv") if ".xlsx" in file_path_curves else file_path_curves
            if not csv_curve_path.endswith(".csv"): csv_curve_path += ".csv"
            df_curves = pd.read_csv(csv_curve_path)
            print(f"成功从 CSV 加载曲线数据: {csv_curve_path}")
        except Exception as e_curves_csv:
            raise FileNotFoundError(
                f"无法从 '{file_path_curves}' 加载曲线数据. CSV 错误: {e_curves_csv}")
    print(f"曲线 (预计) 数据已加载. 列名: {df_curves.columns.tolist()}")

    try:
        df_markers = pd.read_excel(file_path_markers, sheet_name=0)
    except Exception as e_markers_excel:
        print(
            f"提示: 读取 '{file_path_markers}' (Sheet 0) Excel 文件失败 ({e_markers_excel}). 尝试读取 CSV ('coal_data.xlsx - up.csv').")
        try:
            df_markers = pd.read_csv('coal_data.xlsx - up.csv')
            print(f"成功从 CSV 加载标记数据: coal_data.xlsx - up.csv")
        except Exception as e_markers_csv:
            raise FileNotFoundError(
                f"无法从 '{file_path_markers}' 加载标记数据. CSV 错误: {e_markers_csv}")
    print(f"标记 (实际) 数据已加载. 列名: {df_markers.columns.tolist()}")

    key_col_curves = get_key_column(df_curves, prediction_time_key_options)
    key_col_markers = get_key_column(df_markers, prediction_time_key_options)

    if not key_col_curves:
        raise ValueError(f"未在曲线数据中找到预测时间关键列. 列名: {df_curves.columns.tolist()}")
    if not key_col_markers:
        raise ValueError(f"未在标记数据中找到预测时间关键列. 列名: {df_markers.columns.tolist()}")

    for col in q_columns:
        df_curves[col] = pd.to_numeric(df_curves[col], errors='coerce')
        df_markers[col] = pd.to_numeric(df_markers[col], errors='coerce')

    df_curves_cleaned = df_curves.dropna(subset=q_columns).copy()
    if len(df_curves_cleaned) < len(df_curves):
        print(f"提示: 由于Q列存在NaN值，从曲线数据中删除了 {len(df_curves) - len(df_curves_cleaned)} 行。")
    if df_curves_cleaned.empty:
        raise ValueError("在对Q列进行NaN清理后，没有可用的曲线数据。")

    key_col_curves_str = f"{key_col_curves}_str_key_for_merge"
    df_curves_cleaned[key_col_curves_str] = df_curves_cleaned[key_col_curves].astype(str)

    df_markers_subset = df_markers[[key_col_markers] + q_columns].copy()
    key_col_markers_str = f"{key_col_markers}_str_key_for_merge_markers"
    df_markers_subset[key_col_markers_str] = df_markers_subset[key_col_markers].astype(str)

    df_markers_renamed = df_markers_subset.rename(columns={col: f"{col}_marker" for col in q_columns})
    if key_col_markers != key_col_markers_str and key_col_markers in df_markers_renamed.columns:
        df_markers_renamed = df_markers_renamed.drop(columns=[key_col_markers])

    merged_df = pd.merge(
        df_curves_cleaned,
        df_markers_renamed,
        left_on=key_col_curves_str,
        right_on=key_col_markers_str,
        how='left'
    )
    if key_col_markers_str in merged_df.columns:
        merged_df = merged_df.drop(columns=[key_col_markers_str])

    print(f"\n开始绘制Q系列图表. 唯一预测时间数量: {merged_df[key_col_curves_str].nunique()}")

    fig, ax = plt.subplots(figsize=(14, 8))
    unique_prediction_times_str = merged_df[key_col_curves_str].unique()

    legend_handles_list = []
    legend_labels_list = []
    dynamic_handler_map = {}

    for current_pred_time_val_str in unique_prediction_times_str:
        row_data_list = merged_df.loc[merged_df[key_col_curves_str] == current_pred_time_val_str]
        if row_data_list.empty:
            continue
        row_data = row_data_list.iloc[0]

        y_values_curve_original = row_data[q_columns].values.astype(float)
        y_values_curve_negative = -np.abs(y_values_curve_original)

        if len(x_coordinates_q) > 3:
            spline = make_interp_spline(x_coordinates_q, y_values_curve_negative, k=3)
            x_smooth = np.linspace(x_coordinates_q.min(), x_coordinates_q.max(), 300)
            y_smooth = spline(x_smooth)
            line_artist, = ax.plot(x_smooth, y_smooth)
        else:
            line_artist, = ax.plot(x_coordinates_q, y_values_curve_negative, marker='None', linestyle='-')

        current_plot_color = line_artist.get_color()

        line_legend_proxy = Line2D([0], [0], color=current_plot_color,
                                   linestyle=line_artist.get_linestyle(),
                                   lw=line_artist.get_linewidth())

        marker_q_cols_renamed = [f"{col}_marker" for col in q_columns]
        has_actual_data_for_row = not row_data[marker_q_cols_renamed].isnull().all()

        marker_legend_proxy = None
        text2_for_handler = ""
        time_label_part = f"({current_pred_time_val_str}t)"

        if has_actual_data_for_row:
            y_values_marker_original_series = row_data[marker_q_cols_renamed].astype(float)
            valid_marker_indices = ~np.isnan(y_values_marker_original_series)

            if np.any(valid_marker_indices):
                y_values_marker_plot = -np.abs(y_values_marker_original_series[valid_marker_indices])
                x_coords_for_markers_plot = x_coordinates_q[valid_marker_indices]

                if x_coords_for_markers_plot.size > 0:
                    ax.plot(x_coords_for_markers_plot, y_values_marker_plot, marker='.', linestyle='None',
                            markersize=8, color=current_plot_color)
                    marker_legend_proxy = Line2D([0], [0], marker='.', color=current_plot_color,
                                                 linestyle='None', markersize=8)
                    text2_for_handler = f"实际 {time_label_part}"
                else:
                    has_actual_data_for_row = False
            else:
                has_actual_data_for_row = False

        if marker_legend_proxy and has_actual_data_for_row:
            custom_handle = (line_legend_proxy, marker_legend_proxy)
            legend_handles_list.append(custom_handle)
            legend_labels_list.append("")

            handler_instance = HandlerInterspersedText(
                text1_content="预计",
                text2_content=text2_for_handler,
                default_xpad_points=1.5,  # 常规间距
                pad_after_text1_points=10.0,  # “预计”和“点”之间的特定间距 (您可以调整这个值)
                text_color=None
            )
            dynamic_handler_map[custom_handle] = handler_instance
        else:
            legend_handles_list.append(line_legend_proxy)
            legend_labels_list.append(f"预计 {time_label_part}")

    # 设置图表属性
    ax.set_xlabel("指定空间坐标 (Spatial Coordinate for Q-series)")
    ax.set_ylabel("Q系列数值 (下沉 - 负值)\nQ-series Value (Subsidence - Negative)")
    ax.set_title("Q系列下沉数据\nQ-series Subsidence Data")
    ax.grid(False)

    if x_coordinates_q.size > 0:
        x_min_coord = x_coordinates_q.min()
        x_max_coord = x_coordinates_q.max()
        if x_min_coord == x_max_coord:
            ax.set_xticks([x_min_coord])
        else:
            x_min_tick = np.floor(x_min_coord / 40) * 40
            x_max_tick = np.ceil(x_max_coord / 40) * 40
            if x_min_tick <= x_max_tick and (x_max_tick - x_min_tick) / 40 < 20:
                ax.set_xticks(np.arange(x_min_tick, x_max_tick + 1e-9, 40))
            else:
                ax.set_xticks(np.linspace(x_min_coord, x_max_coord, 5).round(1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 显示图例
    num_legend_entries = len(legend_handles_list)
    if num_legend_entries > 0:
        legend_title = "图例"

        handle_length_for_legend = 10.0
        legend_font_size = 10

        if num_legend_entries > 7:
            ax.legend(legend_handles_list, legend_labels_list, title=legend_title,
                      handler_map=dynamic_handler_map,
                      bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize=legend_font_size,
                      handlelength=handle_length_for_legend,
                      numpoints=1,
                      markerscale=0.8,
                      labelspacing=0.8,
                      frameon=False)  # <<< 修改：去掉边框
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            ax.legend(legend_handles_list, legend_labels_list, title=legend_title,
                      handler_map=dynamic_handler_map,
                      loc='best',
                      fontsize=legend_font_size,
                      handlelength=handle_length_for_legend,
                      numpoints=1,
                      markerscale=0.8,
                      labelspacing=0.8,
                      frameon=False)  # <<< 修改：去掉边框
            plt.tight_layout()
    else:
        plt.tight_layout()

    # 再次提醒：请您仔细检查脚本中是否还有任何其他代码可能在图片左下角绘制 “图例：实际 (时间t)” 这行字。
    plt.savefig(plot_filename_q)
    print(f"\n图表已保存为 {plot_filename_q}")

except FileNotFoundError as e:
    print(f"错误: 未找到数据文件。请检查路径。详情: {e}")
except ValueError as e:
    print(f"数据处理错误: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")
    import traceback

    traceback.print_exc()
    print("请确保您的Excel/CSV文件格式正确，并且所有必需的列都存在。")

print("\n--- Q系列绘图脚本结束 ---")