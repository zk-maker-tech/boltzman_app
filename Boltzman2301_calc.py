import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 读取excel文件
file_path = 'coal_data.xlsx'
# Assuming the data is in the first sheet (sheet_name=0) as per your script
try:
    dataframe = pd.read_excel(file_path, sheet_name=0)
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。")
    exit()
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    exit()

# Time values for the FITTING process (from Excel's 2nd column, index 1, as per your script)
t_for_fitting_source = dataframe.iloc[:, 1].copy()

# Time values for the PREDICTION output (from Excel's 4th column, index 3, as per new requirement)
# This will be the x-axis for your calculated Boltzmann curves
time_for_prediction_output_raw = dataframe.iloc[:, 3].copy()
# Convert to NumPy array for consistent use in Boltzmann function and DataFrame creation
time_for_prediction_output_np = time_for_prediction_output_raw.to_numpy() if isinstance(time_for_prediction_output_raw,
                                                                                        pd.Series) else np.asarray(
    time_for_prediction_output_raw)


# Boltzmann时间函数
def Boltzmann(t_arg, A_arg, t0_arg, B_arg):  # Renamed arguments to avoid clashes
    exponent = np.clip((t_arg - t0_arg) / B_arg, -700, 700)  # Prevents overflow in exp
    return -A_arg / (1 + np.exp(exponent)) + A_arg


# R² calculation function
def calculate_r2(y_true_calc, y_pred_calc):
    # Ensure inputs are numpy arrays
    y_true_arr = np.asarray(y_true_calc)
    y_pred_arr = np.asarray(y_pred_calc)

    # Filter out NaNs or Infs that might result from operations
    valid_indices_r2 = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_clean = y_true_arr[valid_indices_r2]
    y_pred_clean = y_pred_arr[valid_indices_r2]

    if len(y_true_clean) == 0:  # No valid data points for R2 calculation
        return np.nan

    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    if ss_tot == 0:  # Avoid division by zero if all true values are the same
        return 1.0 if ss_res < 1e-9 else 0.0  # Perfect fit or no variance
    return 1 - (ss_res / ss_tot)


# List to store fitting parameter summaries (for observation)
fit_summary_list = []
# DataFrame to store the calculated Boltzmann curves using time_for_prediction_output_np
calculated_curves_df = pd.DataFrame({'Prediction_Time': time_for_prediction_output_np})

# Loop through data columns (from Excel's 5th column, index 4, onwards for 'wt' values)
for i in range(4, dataframe.shape[1]):
    wt_original_series = dataframe.iloc[:, i].copy()
    column_name = dataframe.columns[i]
    print(f"\nProcessing column: {column_name}")

    wt_abs_series = np.abs(wt_original_series)

    # --- Data preparation for FITTING ---
    # Align t_for_fitting_source with current wt_abs_series by handling NaNs from EITHER series
    # This ensures t_fit and wt_fit used in curve_fit are clean and aligned.
    combined_for_fit = pd.DataFrame({'t': t_for_fitting_source, 'wt': wt_abs_series}).dropna()
    t_fit_np = combined_for_fit['t'].to_numpy()
    wt_fit_np = combined_for_fit['wt'].to_numpy()

    # --- Robustness checks for FITTING data ---
    if len(t_fit_np) == 0 or len(wt_fit_np) == 0:  # Check after NaN removal
        print(f"  Skipping column {column_name}: No valid data after NaN removal for fitting.")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.nan, '相对误差': np.nan, '备注': '无有效拟合数据'
        })
        continue
    if not np.any(wt_fit_np > 0):
        print(f"  Skipping column {column_name}: All 'wt' values for fitting are zero or negative after NaN removal.")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.nan, '相对误差': np.nan, '备注': 'Wt无正值(拟合用)'
        })
        continue
    if len(t_fit_np) < 3:  # curve_fit needs at least as many points as parameters
        print(
            f"  Skipping column {column_name}: Insufficient data points ({len(t_fit_np)}) for fitting after NaN removal.")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit_np) if len(wt_fit_np) > 0 else np.nan,
            '相对误差': np.nan, '备注': '数据点不足(拟合用)'
        })
        continue

    # --- Initial parameter guess for FITTING ---
    initial_A = np.max(wt_fit_np)
    initial_t0 = np.median(t_fit_np)
    # Ensure t_fit_np has enough elements for std calculation
    initial_B = np.std(t_fit_np) if len(t_fit_np) >= 2 else 1.0
    if initial_B == 0: initial_B = 1e-3  # Avoid B=0

    initial_params_guess = [initial_A, initial_t0, initial_B]
    if pd.Series(initial_params_guess).isnull().any():
        print(f"  Skipping column {column_name}: Initial guess for fitting contains NaN.")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit_np), '相对误差': np.nan, '备注': '初始猜测含NaN'
        })
        continue

    try:
        # --- Perform curve fitting ---
        # Ensure inputs to curve_fit are finite
        if not (np.all(np.isfinite(t_fit_np)) and np.all(np.isfinite(wt_fit_np))):
            raise ValueError("Input arrays for curve_fit contain non-finite values after explicit checks.")

        fitted_params, _ = curve_fit(Boltzmann, t_fit_np, wt_fit_np, p0=initial_params_guess, maxfev=8000)
        A_val, t0_val, B_val = fitted_params
        print(f"  Column {column_name} fitted: A={A_val:.3f}, t0={t0_val:.3f}, B={B_val:.3f}")

        # --- Requirement 3: Calculate Boltzmann function values using fitted A, t0, B ---
        # --- and the 'time_for_prediction_output_np' (from Excel's 4th column) ---
        predicted_curve = Boltzmann(time_for_prediction_output_np, A_val, t0_val, B_val)
        calculated_curves_df[column_name] = predicted_curve  # Add as a new column

        # --- For observation: Calculate R2 and other metrics using FITTING data ---
        # todo pim
        r2_score = calculate_r2(wt_fit_np, Boltzmann(t_fit_np, A_val, t0_val, B_val))
        Wm_observed_fit_data = np.max(wt_fit_np)
        relative_err = (A_val - Wm_observed_fit_data) / Wm_observed_fit_data if Wm_observed_fit_data != 0 else np.nan

        fit_summary_list.append({
            '列名': column_name,
            'A': A_val, 't0': t0_val, 'B': B_val,
            'R2': r2_score, 'Wm_实测_拟合用': Wm_observed_fit_data,
            '相对误差': relative_err, '备注': '拟合成功'
        })

    except RuntimeError:
        print(f"  Fit failed for column {column_name} (RuntimeError).")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit_np) if len(wt_fit_np) > 0 else np.nan,
            '相对误差': np.nan, '备注': '拟合失败 (RuntimeError)'
        })
    except ValueError as ve:
        print(f"  Fit failed for column {column_name} (ValueError: {ve}).")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit_np) if len(wt_fit_np) > 0 else np.nan,
            '相对误差': np.nan, '备注': f'拟合失败 (ValueError)'
        })
    except Exception as e:
        print(f"  An unexpected error occurred for column {column_name}: {e}")
        fit_summary_list.append({
            '列名': column_name, 'A': np.nan, 't0': np.nan, 'B': np.nan,
            'R2': np.nan, 'Wm_实测_拟合用': np.max(wt_fit_np) if len(wt_fit_np) > 0 else np.nan,
            '相对误差': np.nan, '备注': f'拟合失败 ({type(e).__name__})'
        })

# --- Output for observation ---

# Convert summary list to DataFrame for prettier printing
fit_summary_df = pd.DataFrame(fit_summary_list)
print("\n\n--- Fit Summary (for observation) ---")
if not fit_summary_df.empty:
    # Set float format for better readability in print output
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(fit_summary_df)
else:
    print("No fitting attempts were made or summarized.")

print("\n\n--- Calculated Boltzmann Curves using Excel's 4th column as Time Axis (First 5 rows for observation) ---")
if len(calculated_curves_df.columns) > 1:  # Check if any curves were added beyond 'Prediction_Time'
    print(calculated_curves_df.head())
    # (Optional) Save the calculated curves to a new Excel file
    try:
        output_curves_filepath = "calculated_boltzmann_curves_with_col4_time.xlsx"
        calculated_curves_df.to_excel(output_curves_filepath, index=False, float_format="%.3f")
        print(f"\nCalculated Boltzmann curves saved to: {output_curves_filepath}")
    except Exception as e:
        print(f"\nError saving calculated curves to Excel: {e}")
else:
    print("No Boltzmann curves were calculated (e.g., all fits might have failed).")
