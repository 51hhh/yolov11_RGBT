import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

# 深度图归一化参数（单位：米）
MIN_DEPTH = 0.0    # 最小深度，小于这个值的设为0
MAX_DEPTH = 10.0    # 最大深度，大于这个值的设为255
SAVE_DEPTH_SCALE = 255.0 / (MAX_DEPTH - MIN_DEPTH)  # 深度值缩放系数

def normalize_depth(depth_image, depth_scale):
    """将深度图转换为归一化的灰度图（0-255）
    Args:
        depth_image: 原始深度图
        depth_scale: RealSense depth_scale (米/原始深度单位)
    Returns:
        normalized: uint8 灰度图，0表示太近或无效，255表示太远
    """
    # 转换为米为单位
    depth_meters = depth_image * depth_scale

    # 限制在有效范围内
    depth_meters = np.clip(depth_meters, MIN_DEPTH, MAX_DEPTH)

    # 归一化到0-255
    normalized = ((depth_meters - MIN_DEPTH) * SAVE_DEPTH_SCALE).astype(np.uint8)

    # 处理无效值（深度为0的点）
    normalized[depth_image == 0] = 0

    return normalized

def combine_images(rgb_image, depth_image):
    """将RGB图像和深度图像堆叠在一起形成四通道图像"""
    depth_grayscale = normalize_depth(depth_image, depth_scale)
    # 将灰度图像转换为单通道图像并扩展维度
    depth_grayscale = np.expand_dims(depth_grayscale, axis=-1)
    # 堆叠RGB图像和深度图像
    combined_image = np.concatenate((rgb_image, depth_grayscale), axis=-1)
    return combined_image

# 初始化RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# 配置深度和彩色数据流（60Hz）
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# 启动相机
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("深度比例系数:", depth_scale)
print(f"深度图显示范围: {MIN_DEPTH:.1f}m - {MAX_DEPTH:.1f}m")

# 配置对齐
align_to = rs.stream.color
align = rs.align(align_to)

# 创建预览窗口
cv2.namedWindow("预览 (按'q'退出)", cv2.WINDOW_AUTOSIZE)

# 加载YOLOv11模型
model_path = r"D:/robotmaster/2026/RGBD/LLVIP/LLVIP-yolo11n-RGBT-midfusion-15/weights/best.pt"  # 替换为你的模型路径
model = YOLO(model_path)

# 初始化帧率计算
start_time = time.time()
frame_count = 0

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # 获取对齐后的深度帧和彩色帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 转换为numpy数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将深度图转换为伪彩色图
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 合并RGB图像和深度图像形成四通道图像
        combined_image = combine_images(color_image, depth_image)

        # 进行目标检测
        results = model(combined_image, imgsz=640, channels=4, use_simotm="RGBT")

        # 处理检测结果
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = result.names[int(box.cls[0].item())]  # 确保这里正确转换为整数
                conf = box.conf[0]

                # 在RGB图像上绘制检测框和类别标签
                cv2.rectangle(color_image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.putText(color_image, f"{cls} {conf:.2f}",
                            (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 在伪彩色深度图上绘制检测框和类别标签
                cv2.rectangle(depth_colormap, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.putText(depth_colormap, f"{cls} {conf:.2f}",
                            (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 计算帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # 显示帧率
        cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(depth_colormap, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示实时预览（彩色图和伪彩色深度图并排）
        preview = np.hstack((
            color_image,
            depth_colormap
        ))

        cv2.imshow("预览 (按'q'退出)", preview)

        # 'q' 键：退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if frame_count > 0:
        duration = elapsed_time
        fps = frame_count / duration
        print(f"\n录制已终止:")
        print(f"- 总帧数: {frame_count}")
        print(f"- 时长: {duration:.1f} 秒")
        print(f"- 平均帧率: {fps:.1f} FPS")
