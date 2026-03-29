"""
 Michelson 干涉条纹圆心自动定位模块 (基于轮廓识别)
 
 算法参考自 https://github.com/julymiaw/physic 的轮廓识别定位法：
 1. 图像二值化
 2. 提取轮廓
 3. 通过计算轮廓的矩 (Moments) 获取圆心
"""
from __future__ import annotations

import cv2 as cv
import numpy as np
from typing import Optional, Tuple


class CenterDetector:
    def __init__(self):
        self.last_center: Optional[Tuple[float, float]] = None

    def find_center(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        通过轮廓识别寻找圆心：
        1. 二值化 (大津法)
        2. 提取外部轮廓
        3. 计算最大轮廓的重心作为圆心
        """
        if frame.ndim == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 1. 二值化 (Otsu)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # 2. 寻找轮廓
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 3. 找到面积最大的轮廓（通常代表干涉区域的整体外轮廓）
        max_cnt = max(contours, key=cv.contourArea)

        # 4. 计算矩 (Moments)
        M = cv.moments(max_cnt)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            return cx, cy
        
        return None

def visualize_center(frame: np.ndarray, center: Tuple[float, float], detector: Optional[CenterDetector] = None):
    """可视化寻找结果"""
    output = frame.copy()
    if output.ndim == 2:
        output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)
        
    cx, cy = int(center[0]), int(center[1])
    # 画十字准星
    cv.drawMarker(output, (cx, cy), (0, 0, 255), cv.MARKER_CROSS, 30, 2)
    # 画几个同心圆作为参考
    for r in [50, 100, 150]:
        cv.circle(output, (cx, cy), r, (0, 255, 0), 1)
        
    cv.putText(output, f"Center: ({cx}, {cy})", (10, 30), 
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return output

if __name__ == "__main__":
    # 该模块现在作为工具类使用，实时相机测试请通过 integrated_app.py 或 process_image.py 调用
    print("CenterDetector (Contour Mode) 模块已就绪。")

