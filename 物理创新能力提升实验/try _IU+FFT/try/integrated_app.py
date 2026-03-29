# -- coding: utf-8 --
"""
集成控制台 (Integrated App)

整合:
1. 基于 LineProfileDetection (Phase Correlation) 的高灵敏度条纹计数
2. 基于 Hough 变换的稳健圆心定位
3. Thorlabs Kinesis 位移台控制
4. 实时视频显示与数据面板
"""

import sys
import os
import traceback
import site

# 强制添加用户 site-packages 路径 (根据 debug_pythonnet.py 的结果)
# 这一步是为了防止环境路径未正确加载
try:
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        if user_site not in sys.path:
            sys.path.insert(0, user_site) # 插入到最前
    elif isinstance(user_site, list):
        for p in user_site:
            if p not in sys.path:
                sys.path.insert(0, p)
except Exception:
    pass

#Hardcode path just in case site.getusersitepackages returns something else or fails
hardcoded_path = r"C:\Users\struc\AppData\Roaming\Python\Python313\site-packages"
if os.path.exists(hardcoded_path) and hardcoded_path not in sys.path:
    sys.path.insert(0, hardcoded_path)

# 优先导入 pythonnet (clr)，防止与其他 DLL 加载冲突
try:
    import clr
    from System import String
    # print(f"DEBUG: pythonnet loaded successfully. {clr}")
except ImportError as e:
    print(f"Warning: Failed to import pythonnet (clr) at startup: {e}")
    # print("sys.path:", sys.path)

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import sys
import os
import threading
import queue
import logging
import traceback
import numpy as np
import cv2
from ctypes import *
import csv
from datetime import datetime

# 1. 环境与路径配置
if sys.platform == "win32":
    # 添加海康相机 SDK 路径
    sys.path.append(r"D:\\物理\\CCD\\MVS\\Development\\Samples\\Python\\MvImport")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
else:
    sys.path.append("./../../../Python/MvImport")

# 添加本地模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'camera'))
sys.path.append(os.path.dirname(__file__)) # For devices.kinesis_stage

# 2. 导入依赖
try:
    from MvCameraControl_class import *
except ImportError:
    print("错误: 无法导入 MvCameraControl_class，请检查 MVS SDK 路径配置。")
    MvCamera = None

try:
    from devices.kinesis_stage import KinesisStage
except ImportError as e:
    print(f"错误: 无法导入 devices.kinesis_stage: {e}")
    KinesisStage = None

try:
    # 导入优化后的 LineProfileDetection
    from fringe_counter import LineProfileDetection
except ImportError:
    try:
        from camera.fringe_counter import LineProfileDetection
    except ImportError:
        print("错误: 无法导入 fringe_counter.LineProfileDetection。")
        LineProfileDetection = None

try:
    from center_detector import CenterDetector, visualize_center
except ImportError:
    try:
        from camera.center_detector import CenterDetector, visualize_center
    except ImportError:
        CenterDetector = None
        visualize_center = None


# 3. 全局状态
@dataclass
class AppState:
    # 相机/算法
    connected_camera: bool = False
    fringe_count: int = 0
    fringe_status: str = "idle"
    inner_radius: float = 0.0
    ref_wavelength_nm: float = 632.8
    
    # 自动找圆心状态
    auto_center_enabled: bool = False
    detected_center: Optional[Tuple[float, float]] = None

    # 位移台
    connected_stage: bool = False
    stage_serial: str = "26250019"
    stage_position: float = 0.0
    is_moving: bool = False
    stage_direction_sign: int = 0  # +1 增加, -1 减少, 0 静止/未知

    # N1 固定方向（由计数器给出，+吐/-吞/0未定）
    n1_fixed_direction: int = 0

    # 系统
    last_error: str = ""
    status_text: str = "Ready"

    # Manual Setup
    is_setup_mode: bool = False
    manual_roi: Optional[Tuple[int, int, int, int]] = (515, 293, 526, 537)
    manual_center: Optional[Tuple[float, float]] = None

    # Data Logging (Sodium Experiment)
    is_logging: bool = False
    log_data: list = field(default_factory=list) # [(time, pos, phase, visibility)]

SCALE_FACTOR = 1.0 # 假设 Kinesis 驱动已自动处理单位转换

# 4. 位移台控制线程
class StageWorker(threading.Thread):
    def __init__(self, app_state: AppState, command_queue: queue.Queue):
        super().__init__(daemon=True)
        self.state = app_state
        self.queue = command_queue
        self.stage: Optional[KinesisStage] = None
        self.running = True
        self.lock = threading.Lock()
        self.force_stop = threading.Event()

    def run(self):
        while self.running:
            try:
                cmd = self.queue.get(timeout=0.1)
                self._handle_command(cmd)
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"Stage error: {e}")
                self.state.last_error = str(e)

            if self.stage:
                try:
                    with self.lock:
                        pos = self.stage.get_position()
                        moving = self.stage.is_moving
                    pos_mm = pos / SCALE_FACTOR

                    # 推断位移方向（基于位置增量，微小阈值过滤抖动），并保持最近方向供显示
                    last_pos = getattr(self, "_last_pos", None)
                    dir_sign = getattr(self, "_last_dir_sign", 0)
                    last_dir_time = getattr(self, "_last_dir_time", 0.0)
                    delta = 0.0 if last_pos is None else pos_mm - last_pos
                    eps = 1e-6
                    now = time.time()
                    if last_pos is not None and abs(delta) > eps:
                        dir_sign = 1 if delta > 0 else -1
                        last_dir_time = now
                    # 保持最近方向 0.5s，超过则归零
                    hold_ms = 0.5
                    if dir_sign != 0 and (now - last_dir_time) > hold_ms:
                        dir_sign = 0

                    # 写回缓存
                    self._last_pos = pos_mm
                    self._last_dir_sign = dir_sign
                    self._last_dir_time = last_dir_time

                    self.state.stage_position = pos_mm
                    self.state.is_moving = moving
                    # 即便停止，也保留近 0.5s 内的方向用于显示/传递
                    self.state.stage_direction_sign = dir_sign
                except Exception:
                    # 获取状态失败通常意味着设备断开或繁忙
                    pass
            else:
                self.state.connected_stage = False

    def _wait_for_move_complete(self, timeout_ms=30000):
        start = time.time()
        self.force_stop.clear()
        time.sleep(0.1)
        while (time.time() - start) * 1000 < timeout_ms:
            if self.force_stop.is_set():
                with self.lock:
                    self.stage.stop()
                break
            try:
                if self.stage:
                    with self.lock:
                        pos = self.stage.get_position()
                        moving = self.stage.is_moving
                    self.state.stage_position = pos / SCALE_FACTOR
                    if not moving:
                        break
            except:
                break
            time.sleep(0.05)

    def _handle_command(self, cmd):
        op = cmd.get('op')
        if op == 'stop':
            self.force_stop.set()
            if self.stage:
                with self.lock:
                    self.stage.stop()
        elif op == 'connect':
            serial = cmd.get('serial')
            try:
                if self.stage: self.stage.close()
                self.stage = KinesisStage()
                # 尝试连接，自动匹配类型
                self.stage.connect(serial=serial)
                self.state.connected_stage = True
                self.state.stage_serial = serial
                self.state.status_text = f"位移台 {serial} 已连接"
            except Exception as e:
                self.state.last_error = f"连接失败: {e}"
        elif op == 'disconnect':
            if self.stage:
                self.stage.close()
                self.stage = None
            self.state.connected_stage = False
        elif op == 'move_rel':
            dist = cmd.get('dist', 0.0)
            if self.stage:
                self.state.is_moving = True
                try:
                    self.force_stop.clear()
                    with self.lock:
                        self.stage.move_by(dist * SCALE_FACTOR, wait=False)
                    self._wait_for_move_complete()
                except Exception as e:
                     self.state.last_error = f"移动失败: {e}"
                finally:
                    self.state.is_moving = False
        elif op == 'move_abs':
            pos = cmd.get('pos', 0.0)
            if self.stage:
                self.state.is_moving = True
                try:
                    self.force_stop.clear()
                    with self.lock:
                        self.stage.move_to(pos * SCALE_FACTOR, wait=False)
                    self._wait_for_move_complete()
                except Exception as e:
                     self.state.last_error = f"移动失败: {e}"
                finally:
                    self.state.is_moving = False
        elif op == 'set_velocity':
            vel = cmd.get('vel', 0.0)
            if self.stage:
                try:
                    with self.lock:
                        self.stage.set_velocity(vel * SCALE_FACTOR)
                except Exception as e:
                    self.state.last_error = str(e)
        elif op == 'move_continuous':
            direction = cmd.get('dir', 1)
            if self.stage:
                try:
                    with self.lock:
                        self.stage.move_continuous(direction)
                    self.state.is_moving = True
                except Exception as e:
                    self.state.last_error = str(e)
        elif op == 'home':
            if self.stage:
                self.state.is_moving = True
                try:
                    with self.lock:
                       self.stage.home()
                except Exception as e:
                    self.state.last_error = str(e)
                finally:
                    self.state.is_moving = False

# 5. 相机工作线程
class CameraWorker(threading.Thread):
    def __init__(self, app_state: AppState):
        super().__init__(daemon=True)
        self.state = app_state
        self.running = True
        self.cam = None
        self.detector = None
        self.center_detector = CenterDetector() if CenterDetector else None
        if LineProfileDetection:
            # 使用极高灵敏度配置 (针对 0.0005mm/s 慢速优化)
            self.detector = LineProfileDetection(settings={
                "min_signal_strength": 0.01,    # 极低信号门槛
                "min_corr_peak": 0.05,          # 极低相关峰门槛
                "min_abs_delta_fringe": 0.005,  # 极小死区: 捕捉微小蠕动
                "direction_confirm_frames": 1,  # 无需多帧确认，直接累积
                "shift_median_window": 1,       # 关闭滤波延迟，实时响应
                "max_abs_shift_px": 30.0,       # 允许大范围搜索
                "smooth_sigma": 1.0,
                "min_peak_freq": 3              # 兼容稀疏条纹
            })

        # Mouse Interaction State
        self.drag_start = None
        self.drag_mode = None

    def on_mouse(self, event, x, y, flags, param):
        if not self.state.is_setup_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_mode = 'roi'
            self.drag_start = (x, y)
            # Init empty rect if None
            if not self.state.manual_roi:
                 self.state.manual_roi = (x, y, 0, 0)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.state.manual_center = (float(x), float(y))
            self.drag_mode = 'center'
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == 'roi' and self.drag_start:
                x0, y0 = self.drag_start
                # Calculate rect
                rx = min(x0, x)
                ry = min(y0, y)
                rw = abs(x - x0)
                rh = abs(y - y0)
                self.state.manual_roi = (rx, ry, rw, rh)
            elif (flags & cv2.EVENT_FLAG_RBUTTON): # Dragging center
                self.state.manual_center = (float(x), float(y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None
            self.drag_start = None

    def run(self):
        if not MvCamera:
            self.state.last_error = "MVS SDK 未加载"
            return
        
        try:
            deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0 or deviceList.nDeviceNum == 0:
                self.state.last_error = "找不到相机"
                return

            stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            self.cam = MvCamera()
            if self.cam.MV_CC_CreateHandle(stDeviceList) != 0: return
            if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0: return

            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

            self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            self.cam.MV_CC_StartGrabbing()
            self.state.connected_camera = True

            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))
            
            win_name = "Live View"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win_name, self.on_mouse)

            while self.running:
                ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
                if ret == 0:
                    try:
                        buffer_ptr = cast(stOutFrame.pBufAddr, POINTER(c_ubyte * stOutFrame.stFrameInfo.nFrameLen))
                        img_np = np.frombuffer(buffer_ptr.contents, dtype=np.uint8)
                        img_np = img_np.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth)
                        
                        display_img = img_np
                        if self.detector:
                            # Apply Manual Config
                            self.detector.set_manual_config(self.state.manual_roi, self.state.manual_center)

                            if self.state.is_setup_mode:
                                # Setup Mode: Skip heavy algorithm computation to ensure High FPS (low latency cursors)
                                display_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                                
                                # Draw crosshair for mouse center hint
                                h, w = display_img.shape[:2]
                                cv2.line(display_img, (w//2, 0), (w//2, h), (50, 50, 50), 1)
                                cv2.line(display_img, (0, h//2), (w, h//2), (50, 50, 50), 1)
                                
                            else:
                                # Normal Running Mode
                                # 自动圆心检测
                                if self.state.auto_center_enabled and self.center_detector:
                                    detected = self.center_detector.find_center(img_np)
                                    if detected:
                                        self.state.detected_center = detected
                                        # 同步给算法
                                        self.detector.set_manual_config(self.state.manual_roi, detected)

                                # 将位移台状态传递给算法以增强鲁棒性
                                self.detector.set_stage_status(
                                    moving=self.state.is_moving,
                                    direction=self.state.stage_direction_sign,
                                    velocity=0.0,
                                    position=self.state.stage_position
                                )
                                # Pass reference wavelength if set in UI (AppState doesn't store it, retrieve from Main Window logic?)
                                # Actually, cleaner to read it from a global or shared state.
                                # For now, let's just make sure the UI updates a field in AppState or similar.
                                if hasattr(self.state, 'ref_wavelength_nm'):
                                    self.detector.set_ref_wavelength(self.state.ref_wavelength_nm)
                                
                                display_img = self.detector.update(img_np)
                                
                                # 更新状态
                                self.state.fringe_count = self.detector.count_value
                                self.state.fringe_status = self.detector.status or "idle"
                                self.state.inner_radius = getattr(self.detector, 'previous_inner_radius', 0.0)
                                self.state.n1_fixed_direction = getattr(self.detector, '_latched_direction', 0)
                            
                        else:
                            display_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

                        # 叠加 UI 信息
                        color = (0, 0, 255) if self.state.is_moving else (0, 255, 0)
                        status_str = "MOVING" if self.state.is_moving else "IDLE"
                        cv2.putText(display_img, f"Stage: {status_str} @ {self.state.stage_position:.4f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Logging Indicator
                        if self.state.is_logging:
                            pts = len(self.state.log_data)
                            cv2.putText(display_img, f"REC: {pts} pts", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                            
                            # Auto-log data if moving or periodically
                            if self.detector:
                                vis = getattr(self.detector, 'current_visibility', 0.0)
                                self.state.log_data.append((
                                    time.time(), 
                                    self.state.stage_position, 
                                    self.state.fringe_count, 
                                    vis
                                ))

                        # Setup Mode Overlay
                        if self.state.is_setup_mode:
                            h, w = display_img.shape[:2]
                            cv2.rectangle(display_img, (0, 0), (w-1, h-1), (0, 255, 255), 4)
                            cv2.putText(display_img, "SETUP MODE: Drag Left=ROI, Right=Center", (10, h-20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Draw Roi
                            if self.state.manual_roi:
                                rx, ry, rw, rh = self.state.manual_roi
                                cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
                            
                            # Draw Center
                            if self.state.manual_center:
                                cx, cy = self.state.manual_center
                                cv2.circle(display_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                            
                        # Draw detected center if enabled
                        if self.state.auto_center_enabled and self.state.detected_center:
                            cx, cy = self.state.detected_center
                            cv2.drawMarker(display_img, (int(cx), int(cy)), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
                            cv2.putText(display_img, "AUTO CENTER", (int(cx)+10, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        cv2.imshow(win_name, display_img)
                        if cv2.waitKey(1) & 0xFF == 27:
                            pass
                    except Exception:
                        pass
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                else:
                    time.sleep(0.01)

        except Exception as e:
            self.state.last_error = f"Camera Error: {e}"
        finally:
            if self.cam:
                try:
                    self.cam.MV_CC_StopGrabbing()
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
                except: pass
            cv2.destroyAllWindows()
            self.state.connected_camera = False

    def stop(self):
        self.running = False


# 6. 主窗口 UI
class MainWindow:
    def __init__(self):
        self.state = AppState()
        self.stage_queue = queue.Queue()
        self.stage_worker = StageWorker(self.state, self.stage_queue)
        self.stage_worker.start()
        self.camera_worker = None

        self.root = tk.Tk()
        self.root.title("迈克尔逊干涉实验 - 集成控制台")
        self.root.geometry("900x600")

        self._setup_ui()
        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # 顶部连接栏
        conn_frame = tk.LabelFrame(self.root, text="设备连接", padx=5, pady=5)
        conn_frame.pack(fill="x", padx=5, pady=5)
        
        self.btn_cam = tk.Button(conn_frame, text="打开相机", command=self._toggle_cam, bg="#e1e1e1", width=15)
        self.btn_cam.pack(side="left", padx=5)
        
        # Setup Button
        self.btn_setup = tk.Button(conn_frame, text="标定区域", command=self._toggle_setup, bg="#e1e1e1", width=12)
        self.btn_setup.pack(side="left", padx=5)

        self.btn_auto_center = tk.Button(conn_frame, text="自动寻圆心", command=self._toggle_auto_center, bg="#e1e1e1", width=12)
        self.btn_auto_center.pack(side="left", padx=5)
        
        # New Feature Buttons
        self.btn_ai = tk.Button(conn_frame, text="生成AI诊断报告", command=self._generate_ai_report, bg="lightblue", width=15)
        self.btn_ai.pack(side="left", padx=5)
        
        self.btn_log = tk.Button(conn_frame, text="开始数据记录", command=self._toggle_logging, bg="#e1e1e1", width=12)
        self.btn_log.pack(side="left", padx=5)

        tk.Label(conn_frame, text="|  位移台 SN:").pack(side="left", padx=5)
        self.entry_sn = tk.Entry(conn_frame, width=12)
        self.entry_sn.insert(0, "26250019")
        self.entry_sn.pack(side="left")
        self.btn_stage = tk.Button(conn_frame, text="连接位移台", command=self._toggle_stage, bg="#e1e1e1", width=15)
        self.btn_stage.pack(side="left", padx=5)

        # 主要内容区
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧：干涉数据
        left_panel = tk.LabelFrame(main_frame, text="干涉条纹监测", padx=10, pady=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=5)
        
        tk.Label(left_panel, text="累计计数 (N)", font=("Arial", 14)).pack(anchor="w")
        self.lbl_count = tk.Label(left_panel, text="0", font=("Arial", 48, "bold"), fg="#0000AA")
        self.lbl_count.pack(anchor="center", pady=20)
        
        # 波长设置区
        wl_frame = tk.Frame(left_panel)
        wl_frame.pack(fill="x", pady=5)
        tk.Label(wl_frame, text="参考波长 (nm):").pack(side="left")
        self.entry_wl = tk.Entry(wl_frame, width=8)
        self.entry_wl.insert(0, "632.8")
        self.entry_wl.pack(side="left", padx=5)
        self.entry_wl.bind("<Return>", self._update_ref_wl)
        self.entry_wl.bind("<FocusOut>", self._update_ref_wl)

        # 图像平滑设置
        smooth_frame = tk.Frame(left_panel)
        smooth_frame.pack(fill="x", pady=5)
        tk.Label(smooth_frame, text="抗抖动平滑 (ms):").pack(side="left")
        self.entry_smooth = tk.Entry(smooth_frame, width=5)
        self.entry_smooth.insert(0, "0") # Default 0 = Off
        self.entry_smooth.pack(side="left", padx=5)
        self.btn_smooth = tk.Button(smooth_frame, text="Set", command=self._update_smooth, width=5)
        self.btn_smooth.pack(side="left")

        # 饱和计数阈值设置
        sat_frame = tk.Frame(left_panel)
        sat_frame.pack(fill="x", pady=5)
        tk.Label(sat_frame, text="饱和时长 (ms):").pack(side="left")
        self.entry_sat = tk.Entry(sat_frame, width=5)
        self.entry_sat.insert(0, "180")
        self.entry_sat.pack(side="left", padx=5)
        self.btn_sat = tk.Button(sat_frame, text="Set", command=self._update_sat_thresh, width=5)
        self.btn_sat.pack(side="left")

        tk.Label(left_panel, text="当前状态:", font=("Arial", 12)).pack(anchor="w")
        self.lbl_status = tk.Label(left_panel, text="Idle", font=("Arial", 16), fg="gray")
        self.lbl_status.pack(anchor="center", pady=5)

        tk.Frame(left_panel, height=20).pack()
        tk.Label(left_panel, text="内环半径监测 (px):", font=("Arial", 10)).pack(anchor="w")
        self.lbl_radius = tk.Label(left_panel, text="0.0", font=("Consolas", 14))
        self.lbl_radius.pack(anchor="center")
        
        tk.Button(left_panel, text="重置计数", command=self._reset_count, bg="#ffdddd").pack(side="bottom", fill="x", pady=10)


        # 右侧：位移台控制
        right_panel = tk.LabelFrame(main_frame, text="位移台控制 (Thorlabs Kinesis)", padx=10, pady=10)
        right_panel.pack(side="right", fill="both", expand=True, padx=5)

        tk.Label(right_panel, text="当前位置 (mm)", font=("Arial", 12)).pack(anchor="w")
        self.lbl_pos = tk.Label(right_panel, text="0.0000", font=("Consolas", 32, "bold"), fg="darkgreen")
        self.lbl_pos.pack(anchor="center", pady=10)

        # 运动控制区
        ctrl_box = tk.LabelFrame(right_panel, text="Jog Control", padx=5, pady=5)
        ctrl_box.pack(fill="x", pady=10)

        # 1. Step Size Control
        step_cfg_frame = tk.Frame(ctrl_box)
        step_cfg_frame.pack(fill="x", pady=2)
        
        tk.Label(step_cfg_frame, text="Step Size (mm):").grid(row=0, column=0, columnspan=2, sticky="w")
        self.entry_step = tk.Entry(step_cfg_frame, width=10)
        self.entry_step.insert(0, "0.01")
        self.entry_step.grid(row=0, column=2, columnspan=2, padx=5)

        # 6 Direct Move Buttons
        sizes = [0.1, 0.01, 0.001]
        for i, val in enumerate(sizes):
            tk.Button(step_cfg_frame, text=f"-{val}", command=lambda v=val: self._move_rel(-v), width=5).grid(row=1+i, column=0, padx=1, pady=1)
            tk.Button(step_cfg_frame, text=f"+{val}", command=lambda v=val: self._move_rel(v), width=5).grid(row=1+i, column=1, padx=1, pady=1)
            # Label
            tk.Label(step_cfg_frame, text=f"mm").grid(row=1+i, column=2, sticky="w")

        # 2. Mode Selection
        mode_frame = tk.Frame(ctrl_box)
        mode_frame.pack(fill="x", pady=5)
        self.jog_mode = tk.StringVar(value="step")
        tk.Radiobutton(mode_frame, text="Single Step", variable=self.jog_mode, value="step").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Continuous", variable=self.jog_mode, value="cont").pack(side="left", padx=5)

        # 3. Direction Controls (Double Arrows)
        arrow_frame = tk.Frame(ctrl_box)
        arrow_frame.pack(fill="x", pady=10)
        
        # Use simple ASCII arrows
        tk.Button(arrow_frame, text="<<<", command=lambda: self._jog_action(-1), bg="#ddd", font=("Consolas", 12, "bold"), width=8).pack(side="left", padx=5)
        tk.Button(arrow_frame, text="STOP", command=self._stop_stage, bg="red", fg="white", font=("Arial", 10, "bold"), width=8).pack(side="left", padx=5)
        tk.Button(arrow_frame, text=">>>", command=lambda: self._jog_action(1), bg="#ddd", font=("Consolas", 12, "bold"), width=8).pack(side="left", padx=5)

        # 绝对移动
        tk.Frame(right_panel, height=10).pack()
        abs_frame = tk.Frame(right_panel)
        abs_frame.pack(fill="x")
        tk.Label(abs_frame, text="绝对位置:").pack(side="left")
        self.entry_abs = tk.Entry(abs_frame, width=8)
        self.entry_abs.insert(0, "0.00")
        self.entry_abs.pack(side="left", padx=5)
        tk.Button(abs_frame, text="Go Abs", command=self._move_abs_manual).pack(side="left")

        # 速度设置
        tk.Frame(right_panel, height=10).pack()
        spd_frame = tk.Frame(right_panel)
        spd_frame.pack(fill="x")
        tk.Label(spd_frame, text="最大速度:").pack(side="left")
        self.entry_vel = tk.Entry(spd_frame, width=8)
        self.entry_vel.insert(0, "0.1")
        self.entry_vel.pack(side="left", padx=5)
        tk.Button(spd_frame, text="Set (mm/s)", command=self._set_velocity).pack(side="left")

        # 状态栏
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def _schedule_update(self):
        s = self.state
        # 更新相机 UI
        if s.connected_camera:
            self.btn_cam.config(text="关闭相机", bg="#aaffaa")
        else:
            self.btn_cam.config(text="打开相机", bg="#e1e1e1")
            
        if s.is_setup_mode:
            self.btn_setup.config(text="完成标定", bg="yellow")
        else:
            self.btn_setup.config(text="标定区域", bg="#e1e1e1")
            
        if s.is_logging:
            self.btn_log.config(text="停止记录并保存", bg="red", fg="white")
        else:
            self.btn_log.config(text="开始数据记录", bg="#e1e1e1", fg="black")

        self.lbl_count.config(text=str(s.fringe_count))
        self.lbl_status.config(text=s.fringe_status)
        self.lbl_radius.config(text=f"{s.inner_radius:.1f}")

        # 更新位移台 UI
        if s.connected_stage:
            self.btn_stage.config(text="断开位移台", bg="#aaffaa")
            self.lbl_pos.config(text=f"{s.stage_position:.6f}")
        else:
            self.btn_stage.config(text="连接位移台", bg="#e1e1e1")
            self.lbl_pos.config(text="--.----")

        if s.is_moving:
            self.lbl_pos.config(fg="red")
        else:
            self.lbl_pos.config(fg="darkgreen")

        if s.last_error:
            self.status_bar.config(text=f"ERROR: {s.last_error}", fg="red")
        else:
            self.status_bar.config(text=s.status_text, fg="black")

        self.root.after(100, self._schedule_update)

    def _toggle_cam(self):
        if self.camera_worker and self.camera_worker.is_alive():
            self.camera_worker.stop()
            self.camera_worker.join(0.5)
            self.camera_worker = None
        else:
            self.state.last_error = ""
            self.camera_worker = CameraWorker(self.state)
            self.camera_worker.start()

    def _toggle_setup(self):
        self.state.is_setup_mode = not self.state.is_setup_mode
        if not self.state.is_setup_mode:
             # When exiting setup (clicking "Finish Calibration")
             if self.state.manual_roi:
                  print(f"Manual ROI set: {self.state.manual_roi}")
             if self.state.manual_center:
                  print(f"Manual Center set: {self.state.manual_center}")
             
             # Reset fitting data explicitly
             if self.camera_worker and self.camera_worker.detector:
                 self.camera_worker.detector.reset_fit_data()
                 print("Fitting data reset.")
                 # 通知计数算法：标定完成，开启方向观测窗口
                 if hasattr(self.camera_worker.detector, "notify_calibration_done"):
                     self.camera_worker.detector.notify_calibration_done()

    def _toggle_auto_center(self):
        self.state.auto_center_enabled = not self.state.auto_center_enabled
        if self.state.auto_center_enabled:
            self.btn_auto_center.config(bg="lightgreen", text="停止寻圆心")
        else:
            self.btn_auto_center.config(bg="#e1e1e1", text="自动寻圆心")
            self.state.detected_center = None

    def _generate_ai_report(self):
        """Capture state and generate a prompt package for AI."""
        if not self.state.connected_camera or not self.camera_worker:
            messagebox.showerror("Error", "Camera not connected!")
            return
        
        # 1. Create directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(os.path.dirname(__file__), "AI_Reports", ts)
        os.makedirs(report_dir, exist_ok=True)
        
        # 2. Save Image (Get last frame from worker)
        # Accessing private var is hacky but quick for this
        frame = self.camera_worker.detector.last_output_frame if self.camera_worker.detector else None
        img_path = os.path.join(report_dir, "interference_pattern.jpg")
        
        if frame is not None:
            # Use imencode/tofile to match Windows Unicode path (物理 folder)
            try:
                is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                if is_success:
                    im_buf_arr.tofile(img_path)
            except Exception as e:
                print(f"Image save failed: {e}")
                # Fallback to ASCII path if possible or ignore
        else:
            # Fallback grab
            pass 

        # 3. Generate Prompt Text
        s = self.state
        detector = self.camera_worker.detector
        vis = getattr(detector, 'current_visibility', 0.0) if detector else 0.0
        
        prompt_content = f"""
[AI Diagnosis Request]
Timestamp: {ts}
Experiment: Michelson Interferometer Wavelength Measurement

[Current System State]
- Stage Position: {s.stage_position:.6f} mm
- Fringe Count: {s.fringe_count}
- Movement Status: {"Moving" if s.is_moving else "Idle"}
- Signal Visibility (FFT Strength): {vis:.4f} (Ideal > 0.1)
- Reference Wavelength: {s.ref_wavelength_nm} nm

[Task for AI]
1. Analyze the attached interference image 'interference_pattern.jpg'.
2. Check if the concentric rings are clear, well-centered, and have good contrast.
3. Based on the Visibility metric ({vis:.4f}), judge if the optical path alignment is optimal.
4. If Visibility < 0.05, suggest re-aligning the mirrors (M1/M2 tilt).
5. Confirm if the system is ready for precise measurement.
"""
        txt_path = os.path.join(report_dir, "prompt_for_ai.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_content)
            
        messagebox.showinfo("AI Diagnosis", f"Report generated!\nFolder: {report_dir}\n\nPlease upload the image and text file to your AI model.")

    def _toggle_logging(self):
        self.state.is_logging = not self.state.is_logging
        
        if self.state.is_logging:
            self.state.log_data = [] # Clear old buffer
            print("Started Data Logging...")
        else:
            # Stop and Save
            if not self.state.log_data:
                print("No data collected.")
                return
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Sodium_Experiment_Data_{ts}.csv"
            path = os.path.join(os.path.dirname(__file__), filename)
            
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Position_mm", "Fringe_Count", "Visibility_Strength"])
                    writer.writerows(self.state.log_data)
                
                messagebox.showinfo("Data Saved", f"Saved {len(self.state.log_data)} points to:\n{filename}\n\nYou can now plot 'Visibility vs Position' to find the beat period!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def _toggle_stage(self):
        if self.state.connected_stage:
            self.stage_queue.put({'op': 'disconnect'})
        else:
            sn = self.entry_sn.get().strip()
            self.stage_queue.put({'op': 'connect', 'serial': sn})

    def _move_rel(self, dist):
        if not self.state.connected_stage: return
        self.stage_queue.put({'op': 'move_rel', 'dist': dist})

    def _stop_stage(self):
        self.stage_queue.put({'op': 'stop'})

    def _jog_action(self, direction):
        # direction: 1 or -1
        mode = self.jog_mode.get()
        
        if mode == "step":
            try:
                step_sz = float(self.entry_step.get())
                if step_sz <= 0: return
                self._move_rel(step_sz * direction)
            except:
                pass
        elif mode == "cont":
            # Continuous Move
            # Use move_continuous op
            if not self.state.connected_stage: return
            self.stage_queue.put({'op': 'move_continuous', 'dir': direction})

    def _manual_step(self, direction):
        # Legacy method kept just in case, or alias to _jog_action
        self._jog_action(direction)

    def _move_abs_manual(self):
        try:
            val = float(self.entry_abs.get())
            if not self.state.connected_stage: return
            self.stage_queue.put({'op': 'move_abs', 'pos': val})
        except: pass

    def _set_velocity(self):
        try:
            val = float(self.entry_vel.get())
            self.stage_queue.put({'op': 'set_velocity', 'vel': val})
        except: pass

    def _update_ref_wl(self, event=None):
        try:
            val = float(self.entry_wl.get())
            self.state.ref_wavelength_nm = val
        except ValueError:
            pass

    def _update_smooth(self):
        try:
            val = float(self.entry_smooth.get())
            if val < 0: val = 0.0
            if self.camera_worker and self.camera_worker.detector:
                self.camera_worker.detector.set_smoothing_params(val)
        except ValueError:
            pass

    def _update_sat_thresh(self):
        try:
            val = float(self.entry_sat.get())
            if val < 0: val = 0.0
            if self.camera_worker and self.camera_worker.detector:
                self.camera_worker.detector.settings["sat_time_thresh_ms"] = val
                print(f"Saturation time threshold set to {val} ms")
        except ValueError:
            pass

    def _reset_count(self):
        if self.camera_worker and self.camera_worker.detector:
            self.camera_worker.detector.reset_counters()
            self.camera_worker.detector.reset_state()
            # 重新应用高敏设置
            # (LineProfileDetection reset 不会重置 settings, 所以不需要额外操作)

    def _on_close(self):
        if self.camera_worker: self.camera_worker.stop()
        if self.stage_worker: 
            self.stage_worker.running = False
            self.stage_queue.put({})
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = MainWindow()
    app.root.mainloop()
