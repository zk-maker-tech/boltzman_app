import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 基础使用示例：线性函数拟合
# =====================================================================
print("=" * 60)
print("1. 线性函数拟合示例")
print("=" * 60)

# 定义线性函数
def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b

# 生成带噪声的线性数据
np.random.seed(42)
x_linear = np.linspace(0, 10, 50)
y_true_linear = 2.5 * x_linear + 1.2  # 真实参数: a=2.5, b=1.2
noise = np.random.normal(0, 0.5, x_linear.shape)
y_linear = y_true_linear + noise
# 使用 curve_fit 拟合
popt_linear, pcov_linear = curve_fit(linear_func, x_linear, y_linear)
a_fit, b_fit = popt_linear

print(f"真实参数: a=2.5, b=1.2")
print(f"拟合参数: a={a_fit:.3f}, b={b_fit:.3f}")
print(f"参数误差: Δa={abs(a_fit-2.5):.3f}, Δb={abs(b_fit-1.2):.3f}")

# 计算拟合优度
y_pred_linear = linear_func(x_linear, *popt_linear)
print(y_pred_linear)

r2_linear = 1 - np.sum((y_linear - y_pred_linear)**2) / np.sum((y_linear - np.mean(y_linear))**2)
print(f"R² = {r2_linear:.4f}")


# =====================================================================
# 2. 指数函数拟合
# =====================================================================
print("\n" + "=" * 60)
print("2. 指数函数拟合示例")
print("=" * 60)

# y=a⋅e^(b⋅x)+c
def exponential_func(x, a, b, c):
    """指数函数: y = a * exp(b*x) + c"""
    return a * np.exp(b * x) + c


# 生成指数数据
x_exp = np.linspace(0, 2, 30)
y_true_exp = 2.0 * np.exp(1.5 * x_exp) + 0.5
noise_exp = np.random.normal(0, 0.2, x_exp.shape)
y_exp = y_true_exp + noise_exp

# 指数函数拟合通常需要好的初始猜测
initial_guess_exp = [1.0, 1.0, 0.0]

try:
    popt_exp, pcov_exp = curve_fit(exponential_func, x_exp, y_exp, p0=initial_guess_exp)
    a_exp, b_exp, c_exp = popt_exp
    print(f"真实参数: a=2.0, b=1.5, c=0.5")
    print(f"拟合参数: a={a_exp:.3f}, b={b_exp:.3f}, c={c_exp:.3f}")

    y_pred_exp = exponential_func(x_exp, *popt_exp)
    r2_exp = 1 - np.sum((y_exp - y_pred_exp) ** 2) / np.sum((y_exp - np.mean(y_exp)) ** 2)
    print(f"R² = {r2_exp:.4f}")
except Exception as e:
    print(f"拟合失败: {e}")



# =====================================================================
# 3. 高斯函数拟合（带参数边界约束）
# =====================================================================
print("\n" + "=" * 60)
print("3. 高斯函数拟合（带约束）")
print("=" * 60)

def gaussian_func(x, amplitude, mean, std, baseline):
    """高斯函数: y = amplitude * exp(-0.5*((x-mean)/std)²) + baseline"""
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2) + baseline

# 生成高斯数据
x_gauss = np.linspace(-5, 5, 100)
y_true_gauss = gaussian_func(x_gauss, 3.0, 1.0, 1.5, 0.2)
noise_gauss = np.random.normal(0, 0.1, x_gauss.shape)
y_gauss = y_true_gauss + noise_gauss

# 设置参数边界 [amplitude, mean, std, baseline]
# 格式: bounds=([lower_bounds], [upper_bounds])
bounds_gauss = ([0, -np.inf, 0.1, -np.inf],     # 下界
                [np.inf, np.inf, np.inf, np.inf]) # 上界

initial_guess_gauss = [2.5, 0.5, 1.0, 0.0]

popt_gauss, pcov_gauss = curve_fit(
    gaussian_func, x_gauss, y_gauss,
    p0=initial_guess_gauss,
    bounds=bounds_gauss
)

amp_fit, mean_fit, std_fit, base_fit = popt_gauss
print(f"真实参数: amp=3.0, mean=1.0, std=1.5, baseline=0.2")
print(f"拟合参数: amp={amp_fit:.3f}, mean={mean_fit:.3f}, std={std_fit:.3f}, baseline={base_fit:.3f}")

# =====================================================================
# 4. 参数不确定性分析
# =====================================================================
print("\n" + "=" * 60)
print("4. 参数不确定性分析")
print("=" * 60)

# 计算参数标准误差
param_errors = np.sqrt(np.diag(pcov_gauss))
print("参数标准误差:")
print(f"σ_amplitude = {param_errors[0]:.4f}")
print(f"σ_mean = {param_errors[1]:.4f}")
print(f"σ_std = {param_errors[2]:.4f}")
print(f"σ_baseline = {param_errors[3]:.4f}")

# 计算95%置信区间
confidence_level = 0.95
t_value = 1.96  # 对于大样本，95%置信区间对应1.96倍标准误差

print(f"\n{confidence_level * 100}%置信区间:")
param_names = ['amplitude', 'mean', 'std', 'baseline']
for i, (param, error) in enumerate(zip(popt_gauss, param_errors)):
    ci_lower = param - t_value * error
    ci_upper = param + t_value * error
    print(f"{param_names[i]}: [{ci_lower:.4f}, {ci_upper:.4f}]")

# =====================================================================
# 5. 多项式拟合对比
# =====================================================================
print("\n" + "=" * 60)
print("5. 多项式拟合对比")
print("=" * 60)

# 生成非线性数据
x_poly = np.linspace(-2, 2, 50)
y_true_poly = 0.5 * x_poly ** 3 - 2 * x_poly ** 2 + x_poly + 1
noise_poly = np.random.normal(0, 0.2, x_poly.shape)
y_poly = y_true_poly + noise_poly


# 定义不同阶数的多项式
def poly_2(x, a, b, c):
    return a * x ** 2 + b * x + c


def poly_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def poly_4(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


# 拟合不同阶数多项式
models = [
    (poly_2, "2次多项式", 3),
    (poly_3, "3次多项式", 4),
    (poly_4, "4次多项式", 5)
]

results = []
for func, name, n_params in models:
    try:
        popt, pcov = curve_fit(func, x_poly, y_poly)
        y_pred = func(x_poly, *popt)
        r2 = 1 - np.sum((y_poly - y_pred) ** 2) / np.sum((y_poly - np.mean(y_poly)) ** 2)

        # 计算调整R²（考虑参数个数的影响）
        n = len(y_poly)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_params)

        results.append((name, r2, r2_adj, popt))
        print(f"{name}: R² = {r2:.4f}, 调整R² = {r2_adj:.4f}")
    except Exception as e:
        print(f"{name} 拟合failed: {e}")

# =====================================================================
# 6. 权重拟合示例
# =====================================================================
print("\n" + "=" * 60)
print("6. 权重拟合示例")
print("=" * 60)

# 生成具有不同测量精度的数据
x_weight = np.linspace(0, 10, 30)
y_true_weight = 2 * x_weight + 1

# 前半部分数据精度高（小噪声），后半部分精度低（大噪声）
noise_low = np.random.normal(0, 0.1, 15)  # 高精度
noise_high = np.random.normal(0, 0.5, 15)  # 低精度
noise_weight = np.concatenate([noise_low, noise_high])
y_weight = y_true_weight + noise_weight

# 设置权重（精度越高权重越大）
weights = np.concatenate([np.ones(15) * 10, np.ones(15) * 1])  # 高精度数据权重更大

# 无权重拟合
popt_no_weight, _ = curve_fit(linear_func, x_weight, y_weight)
print(f"无权重拟合: a={popt_no_weight[0]:.3f}, b={popt_no_weight[1]:.3f}")

# 有权重拟合
popt_weight, _ = curve_fit(linear_func, x_weight, y_weight, sigma=1 / np.sqrt(weights))
print(f"加权拟合: a={popt_weight[0]:.3f}, b={popt_weight[1]:.3f}")
print(f"真实参数: a=2.0, b=1.0")

# =====================================================================
# 7. 绘制所有结果
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('curve_fit 使用示例', fontsize=16)

# 线性拟合
axes[0, 0].scatter(x_linear, y_linear, alpha=0.6, label='数据点')
axes[0, 0].plot(x_linear, linear_func(x_linear, *popt_linear), 'r-', label=f'拟合线 (R²={r2_linear:.3f})')
axes[0, 0].set_title('线性函数拟合')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 指数拟合
if 'popt_exp' in locals():
    axes[0, 1].scatter(x_exp, y_exp, alpha=0.6, label='数据点')
    axes[0, 1].plot(x_exp, exponential_func(x_exp, *popt_exp), 'r-', label=f'拟合曲线 (R²={r2_exp:.3f})')
    axes[0, 1].set_title('指数函数拟合')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# 高斯拟合
axes[0, 2].scatter(x_gauss, y_gauss, alpha=0.6, label='数据点')
axes[0, 2].plot(x_gauss, gaussian_func(x_gauss, *popt_gauss), 'r-', label='拟合曲线')
axes[0, 2].set_title('高斯函数拟合（带约束）')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 多项式拟合对比
axes[1, 0].scatter(x_poly, y_poly, alpha=0.6, label='数据点')
colors = ['r-', 'g-', 'b-']
for i, (name, r2, r2_adj, popt) in enumerate(results):
    if i < len(models):
        func = models[i][0]
        axes[1, 0].plot(x_poly, func(x_poly, *popt), colors[i],
                        label=f'{name} (R²={r2:.3f})')
axes[1, 0].set_title('多项式拟合对比')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 权重拟合对比
axes[1, 1].scatter(x_weight, y_weight, c=weights, cmap='viridis', alpha=0.6, label='数据点（颜色=权重）')
axes[1, 1].plot(x_weight, linear_func(x_weight, *popt_no_weight), 'r--', label='无权重拟合')
axes[1, 1].plot(x_weight, linear_func(x_weight, *popt_weight), 'b-', label='加权拟合')
axes[1, 1].set_title('权重拟合对比')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 参数不确定性可视化
param_names_short = ['amp', 'mean', 'std', 'base']
axes[1, 2].errorbar(range(len(popt_gauss)), popt_gauss, yerr=param_errors,
                    fmt='o', capsize=5, capthick=2)
axes[1, 2].set_xticks(range(len(popt_gauss)))
axes[1, 2].set_xticklabels(param_names_short)
axes[1, 2].set_title('参数不确定性')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================================
# 8. curve_fit 关键参数总结
# =====================================================================
print("\n" + "=" * 60)
print("curve_fit 关键参数说明")
print("=" * 60)
print("""
curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, 
          check_finite=True, bounds=(-inf, inf), method='lm', jac=None, **kwargs)

主要参数:
- f: 拟合函数，第一个参数必须是自变量
- xdata: 自变量数据 (1-D array)
- ydata: 因变量数据 (1-D array)
- p0: 参数初始猜测值 (array_like, optional)
- sigma: 数据点的标准差，用于加权拟合 (array_like, optional)
- bounds: 参数边界约束 (2-tuple of array_like, optional)
- method: 优化方法 ('lm', 'trf', 'dogbox')

返回值:
- popt: 最优参数估计 (array)
- pcov: 参数协方差矩阵 (2-D array)

使用建议:
1. 非线性函数需要合理的初始猜测 (p0)
2. 使用 bounds 约束参数范围避免非物理解
3. 通过 sigma 参数实现加权拟合
4. 检查 pcov 对角线元素计算参数不确定性
5. 计算 R² 评估拟合质量
""")