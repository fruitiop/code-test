#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抛体运动模拟 (Projectile Motion Simulation)
============================================
对比理想情况（无空气阻力）和实际情况（有空气阻力）下的抛体运动轨迹。

物理背景:
- 理想情况: 只受重力作用，轨迹为抛物线
- 线性阻力: 空气阻力 F_d = -k * v（适合低速/小物体）
- 二次阻力: 空气阻力 F_d = -k * v^2（适合高速/大物体，更接近真实）

运行要求: pip install numpy matplotlib
使用方法: python3 projectile_motion.py
         python3 projectile_motion.py --show  (弹窗显示图表)
"""

import sys

# ========== 设置 matplotlib 后端和中文支持 ==========
# 必须在导入 pyplot 之前设置后端
import matplotlib
if '--show' in sys.argv:
    matplotlib.use('TkAgg')   # 交互式后端，可弹窗显示
else:
    matplotlib.use('Agg')     # 非交互式后端，只保存图片

import numpy as np
import matplotlib.pyplot as plt
import math
import os

# 设置中文字体，按优先级回退
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',      # macOS 苹方（优先）
    'Heiti SC',         # macOS 黑体
    'STHeiti',          # macOS 华文黑体
    'SimHei',           # Windows 黑体
    'Microsoft YaHei',  # Windows 微软雅黑
    'Arial Unicode MS'  # 通用后备
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# ==================== 物理参数设置 ====================
g = 9.8              # 重力加速度 (m/s^2)
m = 1.0              # 物体质量 (kg)
v0 = 50.0            # 初速度 (m/s)
theta_deg = 45.0     # 发射角度 (度)
k_linear = 0.5       # 线性空气阻力系数 (kg/s)
k_quadratic = 0.01   # 二次空气阻力系数 (kg/m)

theta = math.radians(theta_deg)  # 角度转弧度

# 初始速度分量
v0x = v0 * math.cos(theta)
v0y = v0 * math.sin(theta)

print("=" * 60)
print("抛体运动模拟")
print("=" * 60)
print(f"初始速度: {v0} m/s")
print(f"发射角度: {theta_deg} 度")
print(f"初始速度分量: vx={v0x:.2f} m/s, vy={v0y:.2f} m/s")
print(f"重力加速度: {g} m/s^2")
print(f"线性空气阻力系数: {k_linear} kg/s")
print(f"二次空气阻力系数: {k_quadratic} kg/m")
print("=" * 60)


# ==================== 1. 理想抛体运动 (解析解) ====================
def ideal_projectile(v0, theta, g, dt=0.01):
    """
    理想抛体运动（无空气阻力）
    解析解:
      x(t) = v0*cos(theta)*t
      y(t) = v0*sin(theta)*t - 0.5*g*t^2
    """
    # 总飞行时间: 当 y=0 且 t>0 时求解
    t_total = 2 * v0 * math.sin(theta) / g
    t = np.arange(0, t_total + dt, dt)
    
    x = v0 * math.cos(theta) * t
    y = v0 * math.sin(theta) * t - 0.5 * g * t**2
    
    # 最大高度
    t_max_height = v0 * math.sin(theta) / g
    max_height = v0 * math.sin(theta) * t_max_height - 0.5 * g * t_max_height**2
    
    # 射程
    range_total = v0 * math.cos(theta) * t_total
    
    return t, x, y, t_total, max_height, range_total


# ==================== 2. 有线性空气阻力的抛体运动 (数值解) ====================
def linear_drag_projectile(v0, theta, g, m, k, dt=0.001):
    """
    有线性空气阻力的抛体运动
    运动方程:
      dvx/dt = -(k/m) * vx
      dvy/dt = -g - (k/m) * vy
    使用欧拉法进行数值求解
    """
    x, y = [0.0], [0.0]
    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)
    t = [0.0]
    
    while y[-1] >= 0 or len(y) == 1:
        # 计算加速度
        ax = -(k / m) * vx
        ay = -g - (k / m) * vy
        
        # 更新速度
        vx += ax * dt
        vy += ay * dt
        
        # 更新位置
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
        t.append(t[-1] + dt)
        
        # 防止无限循环
        if len(t) > 500000:
            print("警告: 线性阻力模拟达到最大迭代次数")
            break
    
    # 线性插值找到精确的落地点 (y=0)
    if y[-1] < 0 and len(y) > 1:
        alpha = y[-2] / (y[-2] - y[-1])
        x[-1] = x[-2] + alpha * (x[-1] - x[-2])
        y[-1] = 0.0
        t[-1] = t[-2] + alpha * (t[-1] - t[-2])
    
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    
    return t, x, y, t[-1], np.max(y), x[-1]


# ==================== 3. 有二次空气阻力的抛体运动 (数值解) ====================
def quadratic_drag_projectile(v0, theta, g, m, k, dt=0.001):
    """
    有二次空气阻力的抛体运动 (更接近真实高速情况)
    运动方程:
      dvx/dt = -(k/m) * v * vx
      dvy/dt = -g - (k/m) * v * vy
      其中 v = sqrt(vx^2 + vy^2)
    使用欧拉法进行数值求解
    """
    x, y = [0.0], [0.0]
    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)
    t = [0.0]
    
    while y[-1] >= 0 or len(y) == 1:
        v = math.sqrt(vx**2 + vy**2)
        
        # 计算加速度
        ax = -(k / m) * v * vx
        ay = -g - (k / m) * v * vy
        
        # 更新速度
        vx += ax * dt
        vy += ay * dt
        
        # 更新位置
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
        t.append(t[-1] + dt)
        
        # 防止无限循环
        if len(t) > 500000:
            print("警告: 二次阻力模拟达到最大迭代次数")
            break
    
    # 线性插值找到精确的落地点 (y=0)
    if y[-1] < 0 and len(y) > 1:
        alpha = y[-2] / (y[-2] - y[-1])
        x[-1] = x[-2] + alpha * (x[-1] - x[-2])
        y[-1] = 0.0
        t[-1] = t[-2] + alpha * (t[-1] - t[-2])
    
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    
    return t, x, y, t[-1], np.max(y), x[-1]


# ==================== 计算三种情况 ====================
print("\n[1] 计算理想抛体运动（无空气阻力）...")
t_ideal, x_ideal, y_ideal, t_total_ideal, h_max_ideal, range_ideal = ideal_projectile(v0, theta, g)
print(f"   飞行时间: {t_total_ideal:.2f} s")
print(f"   最大高度: {h_max_ideal:.2f} m")
print(f"   射程:     {range_ideal:.2f} m")

print("\n[2] 计算线性空气阻力下的抛体运动...")
t_linear, x_linear, y_linear, t_total_linear, h_max_linear, range_linear = \
    linear_drag_projectile(v0, theta, g, m, k_linear, dt=0.001)
print(f"   飞行时间: {t_total_linear:.2f} s")
print(f"   最大高度: {h_max_linear:.2f} m")
print(f"   射程:     {range_linear:.2f} m")
print(f"   射程减少: {(range_ideal - range_linear):.2f} m ({(1 - range_linear/range_ideal)*100:.1f}%)")

print("\n[3] 计算二次空气阻力下的抛体运动...")
t_quad, x_quad, y_quad, t_total_quad, h_max_quad, range_quad = \
    quadratic_drag_projectile(v0, theta, g, m, k_quadratic, dt=0.001)
print(f"   飞行时间: {t_total_quad:.2f} s")
print(f"   最大高度: {h_max_quad:.2f} m")
print(f"   射程:     {range_quad:.2f} m")
print(f"   射程减少: {(range_ideal - range_quad):.2f} m ({(1 - range_quad/range_ideal)*100:.1f}%)")


# ==================== 4. 绘制结果对比图 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('抛体运动模拟: 理想情况 vs 空气阻力', fontsize=16, fontweight='bold')

# --- 图1: 轨迹对比 ---
ax1 = axes[0, 0]
ax1.plot(x_ideal, y_ideal, 'b-', linewidth=2, label=f'理想情况 (射程={range_ideal:.1f}m)')
ax1.plot(x_linear, y_linear, 'r--', linewidth=2, label=f'线性阻力 (射程={range_linear:.1f}m)')
ax1.plot(x_quad, y_quad, 'g-.', linewidth=2, label=f'二次阻力 (射程={range_quad:.1f}m)')
ax1.set_xlabel('水平距离 x (m)', fontsize=11)
ax1.set_ylabel('高度 y (m)', fontsize=11)
ax1.set_title('运动轨迹对比')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# --- 图2: 高度随时间变化 ---
ax2 = axes[0, 1]
ax2.plot(t_ideal, y_ideal, 'b-', linewidth=2, label='理想情况')
ax2.plot(t_linear, y_linear, 'r--', linewidth=2, label='线性阻力')
ax2.plot(t_quad, y_quad, 'g-.', linewidth=2, label='二次阻力')
ax2.set_xlabel('时间 t (s)', fontsize=11)
ax2.set_ylabel('高度 y (m)', fontsize=11)
ax2.set_title('高度随时间变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 图3: 速度大小随时间变化 ---
ax3 = axes[1, 0]
# 理想情况的速度
v_ideal_speed = np.sqrt((v0 * np.cos(theta))**2 + (v0 * np.sin(theta) - g * t_ideal)**2)

# 线性阻力情况的速度（通过数值差分计算）
v_linear_speed = []
for i in range(len(t_linear) - 1):
    dt_step = t_linear[i+1] - t_linear[i]
    if dt_step > 0:
        v = np.sqrt(((x_linear[i+1]-x_linear[i])/dt_step)**2 + ((y_linear[i+1]-y_linear[i])/dt_step)**2)
        v_linear_speed.append(v)
    else:
        v_linear_speed.append(v_linear_speed[-1] if v_linear_speed else 0)
v_linear_speed.append(v_linear_speed[-1] if v_linear_speed else 0)

# 二次阻力情况的速度
v_quad_speed = []
for i in range(len(t_quad) - 1):
    dt_step = t_quad[i+1] - t_quad[i]
    if dt_step > 0:
        v = np.sqrt(((x_quad[i+1]-x_quad[i])/dt_step)**2 + ((y_quad[i+1]-y_quad[i])/dt_step)**2)
        v_quad_speed.append(v)
    else:
        v_quad_speed.append(v_quad_speed[-1] if v_quad_speed else 0)
v_quad_speed.append(v_quad_speed[-1] if v_quad_speed else 0)

ax3.plot(t_ideal, v_ideal_speed, 'b-', linewidth=2, label='理想情况')
ax3.plot(t_linear, v_linear_speed, 'r--', linewidth=2, label='线性阻力')
ax3.plot(t_quad, v_quad_speed, 'g-.', linewidth=2, label='二次阻力')
ax3.set_xlabel('时间 t (s)', fontsize=11)
ax3.set_ylabel('速度大小 v (m/s)', fontsize=11)
ax3.set_title('速度大小随时间变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- 图4: 数据对比表格 ---
ax4 = axes[1, 1]
ax4.axis('off')

table_data = [
    ['参数', '理想情况', '线性阻力', '二次阻力'],
    ['飞行时间 (s)', f'{t_total_ideal:.2f}', f'{t_total_linear:.2f}', f'{t_total_quad:.2f}'],
    ['最大高度 (m)', f'{h_max_ideal:.2f}', f'{h_max_linear:.2f}', f'{h_max_quad:.2f}'],
    ['射程 (m)', f'{range_ideal:.2f}', f'{range_linear:.2f}', f'{range_quad:.2f}'],
    ['射程减少 (%)', '0.0%', f'{(1-range_linear/range_ideal)*100:.1f}%', f'{(1-range_quad/range_ideal)*100:.1f}%'],
]

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 设置表头样式
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置数据行样式
colors = ['#D9E1F2', '#F2D9D9', '#E2EFDA']
for i in range(1, 5):
    for j in range(4):
        if j == 0:
            table[(i, j)].set_facecolor('#F2F2F2')
            table[(i, j)].set_text_props(weight='bold')
        else:
            table[(i, j)].set_facecolor(colors[j-1])

ax4.set_title('结果对比表', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('projectile_motion_comparison.png', dpi=150, bbox_inches='tight')
print("\n已保存图像: projectile_motion_comparison.png")


# ==================== 5. 不同发射角度的射程分析 ====================
print("\n" + "=" * 60)
print("不同发射角度的射程分析 (理想 vs 二次阻力)")
print("=" * 60)

angles = np.arange(15, 90, 5)
ranges_ideal_angle = []
ranges_quad_angle = []

for angle_deg in angles:
    angle = math.radians(angle_deg)
    # 理想情况
    _, _, _, _, _, r_i = ideal_projectile(v0, angle, g, dt=0.01)
    ranges_ideal_angle.append(r_i)
    # 二次阻力
    _, _, _, _, _, r_q = quadratic_drag_projectile(v0, angle, g, m, k_quadratic, dt=0.005)
    ranges_quad_angle.append(r_q)
    print(f"角度 {angle_deg:2d}度: 理想射程={r_i:6.2f}m, 二次阻力射程={r_q:6.2f}m")

# 绘制射程-角度曲线
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(angles, ranges_ideal_angle, 'bo-', linewidth=2, markersize=6, label='理想情况')
ax.plot(angles, ranges_quad_angle, 'rs--', linewidth=2, markersize=6, label='二次空气阻力')
ax.set_xlabel('发射角度 (度)', fontsize=12)
ax.set_ylabel('射程 (m)', fontsize=12)
ax.set_title(f'射程与发射角度的关系 (初速度 v0={v0}m/s)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 标记最大射程点
max_idx_ideal = int(np.argmax(ranges_ideal_angle))
max_idx_quad = int(np.argmax(ranges_quad_angle))
ax.plot(angles[max_idx_ideal], ranges_ideal_angle[max_idx_ideal], 'bo', markersize=12,
        markerfacecolor='yellow', markeredgecolor='blue', markeredgewidth=2)
ax.plot(angles[max_idx_quad], ranges_quad_angle[max_idx_quad], 'rs', markersize=12,
        markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=2)
ax.annotate(f'理想最大射程\n{angles[max_idx_ideal]}度, {ranges_ideal_angle[max_idx_ideal]:.1f}m',
            xy=(angles[max_idx_ideal], ranges_ideal_angle[max_idx_ideal]),
            xytext=(angles[max_idx_ideal]+10, ranges_ideal_angle[max_idx_ideal]+10),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')
ax.annotate(f'阻力下最大射程\n{angles[max_idx_quad]}度, {ranges_quad_angle[max_idx_quad]:.1f}m',
            xy=(angles[max_idx_quad], ranges_quad_angle[max_idx_quad]),
            xytext=(angles[max_idx_quad]-15, ranges_quad_angle[max_idx_quad]+20),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('range_vs_angle.png', dpi=150, bbox_inches='tight')
print("\n已保存图像: range_vs_angle.png")

# ==================== 6. 能量分析 (新增功能) ====================
print("\n" + "=" * 60)
print("能量分析: 动能、势能与机械能")
print("=" * 60)


def compute_energy(t, x, y, v0, theta, g, m, mode='ideal'):
    """
    计算动能、势能和总机械能
    mode: 'ideal' 用解析式算速度, 'numerical' 用数值差分算速度
    """
    # 势能: Ep = m * g * y
    Ep = m * g * y
    
    if mode == 'ideal':
        # 理想情况: 解析速度
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta) - g * t
        v = np.sqrt(vx**2 + vy**2)
    else:
        # 数值情况: 通过位置差分估算速度
        v = np.zeros_like(t)
        for i in range(len(t) - 1):
            dt_step = t[i+1] - t[i]
            if dt_step > 0:
                v[i] = np.sqrt(((x[i+1]-x[i])/dt_step)**2 + ((y[i+1]-y[i])/dt_step)**2)
            else:
                v[i] = 0
        v[-1] = v[-2] if len(v) > 1 else 0
    
    # 动能: Ek = 0.5 * m * v^2
    Ek = 0.5 * m * v**2
    
    # 总机械能
    E_total = Ek + Ep
    
    return Ek, Ep, E_total


# 计算三种情况的能量
Ek_ideal, Ep_ideal, E_total_ideal = compute_energy(
    t_ideal, x_ideal, y_ideal, v0, theta, g, m, mode='ideal'
)
Ek_linear, Ep_linear, E_total_linear = compute_energy(
    t_linear, x_linear, y_linear, v0, theta, g, m, mode='numerical'
)
Ek_quad, Ep_quad, E_total_quad = compute_energy(
    t_quad, x_quad, y_quad, v0, theta, g, m, mode='numerical'
)

# 绘制能量分析图
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle('能量分析: 动能、势能与机械能随时间变化', fontsize=14, fontweight='bold')

# 子图1: 动能
ax_ek = axes3[0]
ax_ek.plot(t_ideal, Ek_ideal, 'b-', linewidth=2, label='理想情况')
ax_ek.plot(t_linear, Ek_linear, 'r--', linewidth=2, label='线性阻力')
ax_ek.plot(t_quad, Ek_quad, 'g-.', linewidth=2, label='二次阻力')
ax_ek.set_xlabel('时间 t (s)', fontsize=11)
ax_ek.set_ylabel('动能 Ek (J)', fontsize=11)
ax_ek.set_title('动能变化')
ax_ek.legend()
ax_ek.grid(True, alpha=0.3)

# 子图2: 势能
ax_ep = axes3[1]
ax_ep.plot(t_ideal, Ep_ideal, 'b-', linewidth=2, label='理想情况')
ax_ep.plot(t_linear, Ep_linear, 'r--', linewidth=2, label='线性阻力')
ax_ep.plot(t_quad, Ep_quad, 'g-.', linewidth=2, label='二次阻力')
ax_ep.set_xlabel('时间 t (s)', fontsize=11)
ax_ep.set_ylabel('势能 Ep (J)', fontsize=11)
ax_ep.set_title('势能变化')
ax_ep.legend()
ax_ep.grid(True, alpha=0.3)

# 子图3: 总机械能
ax_et = axes3[2]
ax_et.plot(t_ideal, E_total_ideal, 'b-', linewidth=2, label='理想情况')
ax_et.plot(t_linear, E_total_linear, 'r--', linewidth=2, label='线性阻力')
ax_et.plot(t_quad, E_total_quad, 'g-.', linewidth=2, label='二次阻力')
ax_et.set_xlabel('时间 t (s)', fontsize=11)
ax_et.set_ylabel('总机械能 E (J)', fontsize=11)
ax_et.set_title('总机械能变化')
ax_et.legend()
ax_et.grid(True, alpha=0.3)

# 在总机械能图上标注能量耗散率
E0 = 0.5 * m * v0**2  # 初始总能量
if len(E_total_linear) > 0 and E0 > 0:
    loss_linear = (E0 - E_total_linear[-1]) / E0 * 100
    ax_et.text(0.55, 0.85, f'线性阻力耗散: {loss_linear:.1f}%',
               transform=ax_et.transAxes, fontsize=9, color='red',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
if len(E_total_quad) > 0 and E0 > 0:
    loss_quad = (E0 - E_total_quad[-1]) / E0 * 100
    ax_et.text(0.55, 0.72, f'二次阻力耗散: {loss_quad:.1f}%',
               transform=ax_et.transAxes, fontsize=9, color='green',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('energy_analysis.png', dpi=150, bbox_inches='tight')
print("已保存图像: energy_analysis.png")

print(f"\n初始总机械能: {E0:.2f} J")
print(f"理想情况落地时总能量: {E_total_ideal[-1]:.2f} J (守恒)")
print(f"线性阻力落地时总能量: {E_total_linear[-1]:.2f} J (耗散 {(E0-E_total_linear[-1]):.2f} J)")
print(f"二次阻力落地时总能量: {E_total_quad[-1]:.2f} J (耗散 {(E0-E_total_quad[-1]):.2f} J)")

# ==================== 7. 分析结论 ====================
print("\n" + "=" * 60)
print("分析结论:")
print(f"1. 理想情况下，最大射程在 45 度角度时达到")
print(f"2. 有空气阻力时，最佳发射角度会降低到约 {angles[max_idx_quad]} 度")
print(f"3. 空气阻力会显著减小射程和最大高度")
print(f"4. 二次阻力（高速情况）比线性阻力影响更大")
print(f"5. 理想情况机械能守恒，有阻力时机械能被耗散为热能")
print("=" * 60)

# 根据参数决定是否弹窗显示
if '--show' in sys.argv:
    plt.show()
    print("已显示图表窗口（关闭窗口后程序退出）")
else:
    print("\n提示: 添加 --show 参数可弹窗显示图表")
    print("      示例: python3 projectile_motion.py --show")
