import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

# --- UKF 库导入 ---
from filterpy.kalman import UnscentedKalmanFilter as UKF
# 将 MerweScaledSigmaPoints 移动到 filterpy.kalman 模块导入
from filterpy.kalman import MerweScaledSigmaPoints 
from filterpy.common import Q_discrete_white_noise # Q_discrete_white_noise 仍在 common 中
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# --- 物理常数定义 (请根据实际情况调整) ---
BALL_DIAMETER_M = 0.2275    # 排球直径 (米)
BALL_MASS_KG = 0.270        # 排球质量 (千克)
AIR_DENSITY_RHO = 1.225     # 空气密度 (kg/m^3)
G_ACCEL = 9.81              # 重力加速度 (m/s^2)
BALL_AREA_A = np.pi * (BALL_DIAMETER_M / 2)**2
# 阻力系数常数项 C'_D = (1/2 * rho * A) / m
C_PRIME_D = 0.5 * AIR_DENSITY_RHO * BALL_AREA_A / BALL_MASS_KG

# 深度图归一化参数（单位：米）
MIN_DEPTH = 0.0     # 最小深度
MAX_DEPTH = 10.0    # 最大深度
SAVE_DEPTH_SCALE = 255.0 / (MAX_DEPTH - MIN_DEPTH)

# --- 全局变量 ---
color_intrinsics = None 
depth_scale = 0.0
ukf = None
DT = 0.0 # 帧间隔时间

# 轨迹存储
observed_trajectory = [] 
filtered_trajectory = [] 
predicted_trajectory = [] 

# 3D 可视化全局变量
fig = None
ax = None
scat_obs = None
line_filtered = None
line_predicted = None

# --- 1. UKF 核心函数 (状态转移与测量) ---

def ball_dynamics_dot(x):
    """
    状态导数计算 (x_dot)，包含重力和空气阻力。
    x = [x, y, z, vx, vy, vz, Cd]
    """
    _, _, _, vx, vy, vz, Cd = x
    
    # 速度大小
    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # 阻力加速度 a_drag = - C'_D * Cd * |v| * v
    a_drag_x = -C_PRIME_D * Cd * v_abs * vx
    a_drag_y = -C_PRIME_D * Cd * v_abs * vy
    a_drag_z = -C_PRIME_D * Cd * v_abs * vz
    
    # 总加速度 (a = g + a_drag)
    ax = a_drag_x
    ay = a_drag_y
    az = a_drag_z - G_ACCEL # 假设 z 轴向上
    
    dCd_dt = 0.0 # Cd 假定为恒定
    
    x_dot = np.array([vx, vy, vz, ax, ay, az, dCd_dt])
    return x_dot

def fx(x, dt):
    """
    状态转移函数 f(x_k, dt) - 使用 RK4 积分
    """
    # 4 阶龙格-库塔法 (RK4)
    k1 = dt * ball_dynamics_dot(x)
    k2 = dt * ball_dynamics_dot(x + 0.5 * k1)
    k3 = dt * ball_dynamics_dot(x + 0.5 * k2)
    k4 = dt * ball_dynamics_dot(x + k3)
    
    x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x_next

def hx(x):
    """
    测量函数 h(x) - 仅测量位置 [x, y, z]
    """
    return x[:3]

# --- 2. UKF 鲁棒性改进函数 ---

def R_function(Z_obs):
    """
    动态计算测量噪声协方差 R 矩阵，基于深度 Z (米)。
    【重要改进：动态R矩阵】
    """
    # 静态参考深度下的标准差 (例如 Z=1.0m 处)
    # 这些值需要通过实验标定！
    SIGMA_REF = np.array([0.005, 0.005, 0.015]) # [sigma_x, sigma_y, sigma_z] 在 1m 处的标准差
    REF_Z = 1.0 
    
    # 设置 Z 轴测量噪声的下限，防止 Z 趋近 0 导致 R 趋近 0
    Z_min = 0.5 
    adjusted_Z = max(Z_obs, Z_min)
    
    # 动态方差：方差与深度平方成正比 R(Z) = R_ref * (Z / Z_ref)^2
    sigma_dynamic = SIGMA_REF * (adjusted_Z / REF_Z)
    
    R = np.diag(sigma_dynamic**2)
    return R

def initialize_ukf_filter(dt):
    """初始化全局 UKF 实例"""
    global ukf, DT
    DT = dt
    
    n = 7 # 状态维度
    m = 3 # 测量维度
    
    # Merwe Scaled Sigma Points (n=7, alpha=.1, beta=2., kappa=0 推荐)
    points = MerweScaledSigmaPoints(n=n, alpha=.1, beta=2., kappa=0) 

    ukf = UKF(dim_x=n, dim_z=m, dt=dt, fx=fx, hx=hx, points=points)

    # 初始状态 x_0: 假设静止，Cd=0.4
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]) 

    # 初始协方差 P_0: 对初始状态的不确定性
    ukf.P = np.diag([
        0.5**2, 0.5**2, 0.5**2, # 位置误差
        1.0**2, 1.0**2, 1.0**2, # 速度误差
        5.0**2                  # Cd 误差非常大，允许快速学习
    ])

    # 过程噪声协方差 Q
    # 位置/速度：块对角矩阵，反映积分和未建模动力学噪声
    ukf.Q[:6, :6] = Q_discrete_white_noise(dim=3, dt=dt, var=0.01, block_size=2)
    # Cd 的过程噪声：【重要改进：调大Q】允许Cd适应排球自旋等变化
    ukf.Q[6, 6] = 0.1**2  

    # 初始 R 矩阵 (假设初始深度 1m)
    ukf.R = R_function(1.0)
    
def predict_future_trajectory(initial_state, steps, dt):
    """根据 UKF 当前估计的状态预测未来轨迹"""
    trajectory = []
    x = initial_state.copy()
    
    # 预测未来 10 步 (~0.16秒)
    for _ in range(steps):
        x = fx(x, dt)
        trajectory.append(x[:3])
    return np.array(trajectory)

# --- 3. 图像处理与解算辅助函数 ---

def normalize_depth(depth_image):
    """将深度图转换为归一化的灰度图（0-255）"""
    global depth_scale
    depth_meters = depth_image * depth_scale
    depth_meters = np.clip(depth_meters, MIN_DEPTH, MAX_DEPTH)
    normalized = ((depth_meters - MIN_DEPTH) * SAVE_DEPTH_SCALE).astype(np.uint8)
    normalized[depth_image == 0] = 0
    return normalized

def combine_images(rgb_image, depth_image):
    """将RGB图像和归一化深度图像堆叠在一起形成四通道图像 (用于RGBT YOLO)"""
    depth_grayscale = normalize_depth(depth_image)
    depth_grayscale = np.expand_dims(depth_grayscale, axis=-1)
    combined_image = np.concatenate((rgb_image, depth_grayscale), axis=-1)
    return combined_image

def pixel_to_point(u, v, depth_value_raw, intrinsics):
    """
    使用相机内参将像素坐标和原始深度值解算为相机坐标系下的XYZ坐标（米）
    【关键解算功能】
    """
    global depth_scale
    
    if depth_value_raw <= 0:
        return None

    # Z坐标（深度）
    Z = depth_value_raw * depth_scale
    
    # 获取内参
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # 解算 X 和 Y
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy
    
    return (X, Y, Z)

def get_robust_depth(depth_image, bbox, side_bias=0.1):
    """
    鲁棒地获取检测框内的平均/中位深度值，处理边界和无效深度。
    【重要：目标深度正确获取】
    """
    x1, y1, x2, y2 = bbox
    
    # 限制ROI在图像尺寸内
    h, w = depth_image.shape
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [w - 1, h - 1, w - 1, h - 1])
    
    # 缩小ROI：排除边界区域
    dx = int((x2 - x1) * side_bias)
    dy = int((y2 - y1) * side_bias)
    
    roi = depth_image[y1+dy:y2-dy, x1+dx:x2-dx]
    
    valid_depths = roi[roi > 0]
    
    if valid_depths.size == 0:
        # 如果缩小区域无效，尝试整个框的有效深度
        full_roi = depth_image[y1:y2, x1:x2]
        valid_depths = full_roi[full_roi > 0]
        if valid_depths.size == 0:
             return 0, (x1 + x2) // 2, (y1 + y2) // 2 # 无法获得有效深度
    
    # 使用中位数比平均值更鲁棒
    avg_depth = np.median(valid_depths) 
    
    # 计算像素中心点
    center_u = (x1 + x2) // 2
    center_v = (y1 + y2) // 2
    
    return avg_depth, center_u, center_v

# --- 4. 3D 可视化函数 ---

def initialize_3d_plot():
    """初始化 3D 绘图窗口"""
    global fig, ax, scat_obs, line_filtered, line_predicted
    
    plt.ion() # 开启交互模式
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("排球 3D 轨迹追踪与预测")
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米, 深度)')
    
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.set_zlim([0.5, 5.0])
    
    scat_obs = ax.scatter([], [], [], c='r', marker='o', s=10, label='观测 (XYZ)')
    line_filtered, = ax.plot([], [], [], 'b-', label='UKF 滤波轨迹')
    line_predicted, = ax.plot([], [], [], 'g--', label='UKF 预测轨迹')
    
    ax.legend()
    plt.show()

def update_3d_plot(current_z_obs):
    """更新 3D 轨迹图"""
    global scat_obs, line_filtered, line_predicted, observed_trajectory, filtered_trajectory, predicted_trajectory
        
    # 1. 更新观测轨迹 (最近 50 帧)
    obs_array = np.array(observed_trajectory[-50:]) 
    if obs_array.size > 0:
        scat_obs._offsets3d = (obs_array[:, 0], obs_array[:, 1], obs_array[:, 2])
        
    # 2. 更新滤波轨迹 (全部)
    filtered_array = np.array(filtered_trajectory)
    if filtered_array.size > 0:
        line_filtered.set_data(filtered_array[:, 0], filtered_array[:, 1])
        line_filtered.set_3d_properties(filtered_array[:, 2])
        
    # 3. 更新预测轨迹
    predicted_array = np.array(predicted_trajectory)
    if predicted_array.size > 0:
        line_predicted.set_data(predicted_array[:, 0], predicted_array[:, 1])
        line_predicted.set_3d_properties(predicted_array[:, 2])

    # 动态调整 Z 轴显示
    if current_z_obs and current_z_obs > 0:
        ax.set_zlim([max(0.5, current_z_obs - 1.0), max(5.0, current_z_obs + 1.0)])
    
    fig.canvas.draw()
    fig.canvas.flush_events()


# --- 5. 主程序执行块 ---

# 初始化RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# 配置数据流 (假设 640x480, 60 FPS)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# 加载YOLO模型
# !!! 请替换为你的模型路径 !!!
model_path = r"D:/robotmaster/2026/RGBD/LLVIP/LLVIP-yolo11n-RGBT-midfusion-16/weights/best.pt"  
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"!!! 错误：无法加载 YOLO 模型。请检查路径：{model_path}")
    print(e)
    sys.exit(1)

# 启动相机
try:
    profile = pipeline.start(config)
except Exception as e:
    print(f"!!! 错误：无法启动 RealSense 相机。请检查设备连接。")
    print(e)
    sys.exit(1)


# 获取相机参数
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

frame_rate = color_profile.fps()
DT = 1.0 / frame_rate
initialize_ukf_filter(DT) # 初始化 UKF

# 配置对齐
align_to = rs.stream.color
align = rs.align(align_to)

# 初始化 3D 绘图
initialize_3d_plot()

# 创建预览窗口
cv2.namedWindow("预览 (按'q'退出)", cv2.WINDOW_AUTOSIZE)

print(f"\n系统启动成功：帧率={frame_rate} FPS, DT={DT:.4f} s")

start_time = time.time()
frame_count = 0
last_detection_time = 0.0

try:
    while True:
        current_time = time.time()
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 将深度图转换为伪彩色图
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        combined_image = combine_images(color_image, depth_image)
        results = model(combined_image, imgsz=640, channels=4, use_simotm="RGBT", verbose=False) 
        
        display_color = color_image.copy()
        display_depth = depth_colormap.copy()
        
        is_ball_detected_this_frame = False
        current_z_obs = 0.0

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)  
                cls = result.names[int(box.cls[0].item())]
                conf = box.conf[0]
                
                # *** 确保类别名正确，否则无法进入 UKF 逻辑 ***
                if cls != 'volleyball' and cls != 'ball': 
                    continue 
                
                # --- 1. 获取鲁棒深度和中心点 ---
                avg_depth_raw, center_u, center_v = get_robust_depth(depth_image, r, side_bias=0.1)
                
                # --- 2. 3D 坐标解算 ---
                point_xyz = pixel_to_point(center_u, center_v, avg_depth_raw, color_intrinsics)
                
                if point_xyz and point_xyz[2] > 0:
                    is_ball_detected_this_frame = True
                    last_detection_time = current_time
                    z_k = np.array(point_xyz) # 原始观测值
                    current_z_obs = z_k[2]
                    
                    # --- 3. UKF 预测和更新 (检测到) ---
                    # 动态调整 R 矩阵
                    ukf.R = R_function(z_k[2]) 
                    ukf.predict()
                    ukf.update(z_k)
                    
                    # 轨迹存储与预测
                    observed_trajectory.append(z_k)
                    filtered_trajectory.append(ukf.x[:3].copy())
                    predicted_trajectory = predict_future_trajectory(ukf.x, 20, DT) # 预测未来 20 步
                    
                    # --- 4. 图像信息显示 ---
                    X_f, Y_f, Z_f = ukf.x[:3]
                    Vx_f, Vy_f, Vz_f = ukf.x[3:6]
                    Cd_f = ukf.x[6]
                    
                    xyz_text = f"F_XYZ: {X_f:.2f} {Y_f:.2f} {Z_f:.2f} m"
                    vel_text = f"F_Vel: {Vx_f:.1f} {Vy_f:.1f} {Vz_f:.1f} m/s"
                    cd_text = f"F_Cd: {Cd_f:.3f}"
                    
                    # 绘制
                    cv2.rectangle(display_color, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                    cv2.circle(display_color, (center_u, center_v), 5, (0, 0, 255), -1) 
                    cv2.putText(display_color, xyz_text, (r[0], r[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display_color, vel_text, (r[0], r[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display_color, cd_text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                else:
                    # 深度无效
                    text = f"{cls} {conf:.2f} | 深度无效"
                    cv2.putText(display_color, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        # --- UKF 仅预测 (目标丢失) ---
        if not is_ball_detected_this_frame and filtered_trajectory and (current_time - last_detection_time) < 1.0: # 1秒内丢失
            ukf.predict()
            filtered_trajectory.append(ukf.x[:3].copy())
            predicted_trajectory = predict_future_trajectory(ukf.x, 20, DT)
            # 可以在图像上显示 "Tracking Lost, Predicting..."
            cv2.putText(display_color, "Tracking Lost, Predicting...", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif (current_time - last_detection_time) >= 1.0:
            # 丢失超过 1 秒，清空预测轨迹
            predicted_trajectory = []


        # --- 5. 更新 3D 可视化 ---
        update_3d_plot(current_z_obs)

        # 图像显示
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(display_color, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        preview = np.hstack((display_color, cv2.cvtColor(display_depth, cv2.COLOR_BGR2RGB)))
        cv2.imshow("预览 (按'q'退出)", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if fig:
        plt.close(fig) 
    
    duration = time.time() - start_time
    print(f"\n程序已终止，平均帧率: {frame_count / duration:.1f} FPS")