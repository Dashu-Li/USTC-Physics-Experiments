# -- coding: utf-8 --

import sys
import threading
import os
import time
from ctypes import *
import numpy
import cv2


_WINDOW_HWND: dict[str, int] = {}


def _set_window_title_unicode(window_name: str, title: str) -> None:
    """On Windows, force-set a Unicode window caption for an OpenCV HighGUI window.

    OpenCV's window titles can be garbled under some codepages. We keep an ASCII
    internal window name (used by cv2.imshow) and set the displayed title via Win32.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.SetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        user32.SetWindowTextW.restype = ctypes.c_bool

        hwnd = _WINDOW_HWND.get(window_name)
        if not hwnd:
            # At creation time, the caption equals window_name.
            hwnd = int(user32.FindWindowW(None, window_name) or 0)
            if hwnd:
                _WINDOW_HWND[window_name] = hwnd
        if hwnd:
            user32.SetWindowTextW(ctypes.c_void_p(hwnd), title)
    except Exception:
        return


def _named_window(window_name: str, title: str, flags: int) -> None:
    cv2.namedWindow(window_name, flags)
    _set_window_title_unicode(window_name, title)

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PIL_AVAILABLE = False

from fringe_counter import LineProfileDetection

if sys.platform == 'win32':
    import msvcrt
    sys.path.append(r"D:\\物理\\CCD\\MVS\\Development\\Samples\\Python\\MvImport")
else:
    import termios, tty
    # Demo 目录:  "/opt/xxx/Samples/平台/OpenCV/ZZ/ ")   python接口目录： "/opt/xxx/Samples/平台/Python/MvImport"
    sys.path.append("./../../../Python/MvImport")


from MvCameraControl_class import *
g_bExit = False

# 半径-时间监控窗口的全局状态
radius_recording = False
radius_samples = []  # [(t, r)]，t 为相对起始时间（秒），r 为半径（像素）
radius_record_start: float | None = None
radius_avg_value: float | None = None

# 按钮区域（在“半径监控”窗口中的坐标）
BTN_X1, BTN_Y1, BTN_X2, BTN_Y2 = 20, 230, 200, 270

_FONT_CACHE = {}
_PIL_WARNED = False


def _get_chinese_font(font_size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    font_size = int(max(10, font_size))
    cached = _FONT_CACHE.get(font_size)
    if cached is not None:
        return cached

    # Common Chinese fonts on Windows
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",  # Microsoft YaHei
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",  # SimHei
        r"C:\\Windows\\Fonts\\simsun.ttc",  # SimSun
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                _FONT_CACHE[font_size] = font
                return font
            except Exception:
                pass

    # Fallback to default PIL font (may not support Chinese)
    font = ImageFont.load_default()
    _FONT_CACHE[font_size] = font
    return font


def _draw_text_pil_bgr(img_bgr: numpy.ndarray, items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]]) -> None:
    """Draw multiple text items onto a BGR image using PIL for Chinese support."""
    global _PIL_WARNED
    if not _PIL_AVAILABLE:
        if not _PIL_WARNED:
            print("[UI] 未检测到 Pillow，OpenCV 窗口内中文可能无法正常显示。可运行: pip install pillow")
            _PIL_WARNED = True
        # Best-effort fallback: OpenCV will likely not render Chinese.
        for text, (x, y), _, color in items:
            cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        return

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for text, (x, y), font_size, bgr in items:
        font = _get_chinese_font(font_size)
        # PIL uses RGB
        fill = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text((int(x), int(y)), text, font=font, fill=fill)
    img_bgr[:] = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGB2BGR)


def on_radius_window_mouse(event, x, y, flags, param):
    """半径监控窗口的鼠标回调：点击按钮区域开始/结束记录。"""
    global radius_recording, radius_samples, radius_record_start, radius_avg_value
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            if not radius_recording:
                # 开始记录：清空历史，重置起始时间和平均值
                radius_recording = True
                radius_samples = []
                radius_record_start = None
                radius_avg_value = None
                print("[半径监控] 开始记录离中心最近条纹半径")
            else:
                # 结束记录：计算平均值
                radius_recording = False
                if radius_samples:
                    radii_only = [r for _, r in radius_samples]
                    radius_avg_value = sum(radii_only) / len(radii_only)
                    print(f"[半径监控] 本段平均半径: {radius_avg_value:.2f} 像素")
                else:
                    radius_avg_value = None
                    print("[半径监控] 本段没有有效半径数据")


def update_radius_window(inner_radius: float | None):
    """更新“半径-时间”窗口：绘制曲线和按钮，并在需要时追加新样本。"""
    global radius_recording, radius_samples, radius_record_start, radius_avg_value

    # 创建画布
    height, width = 300, 600
    panel = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    panel[:] = (40, 40, 40)

    # 坐标系区域
    margin_left, margin_right = 60, 20
    margin_top, margin_bottom = 30, 80
    plot_x1, plot_y1 = margin_left, margin_top
    plot_x2, plot_y2 = width - margin_right, height - margin_bottom

    # 追加样本（只在记录状态且半径有效且半径>=50像素时）
    if radius_recording and inner_radius is not None and inner_radius >= 50.0:
        now = time.time()
        if radius_record_start is None:
            radius_record_start = now
        t = now - radius_record_start
        radius_samples.append((t, float(inner_radius)))

    # 绘制坐标轴
    cv2.rectangle(panel, (plot_x1, plot_y1), (plot_x2, plot_y2), (100, 100, 100), 1)
    cv2.putText(panel, "t (s)", (plot_x2 - 60, plot_y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(panel, "r (px)", (10, plot_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 绘制半径-时间曲线
    if len(radius_samples) >= 2:
        times = [t for t, _ in radius_samples]
        radii = [r for _, r in radius_samples]
        t_max = max(times)
        r_min, r_max = min(radii), max(radii)
        if t_max <= 0:
            t_max = 1.0
        if r_max - r_min < 1e-3:
            # 避免除零：如果几乎不变，就给一个小范围
            r_min -= 1.0
            r_max += 1.0

        plot_w = plot_x2 - plot_x1
        plot_h = plot_y2 - plot_y1
        pts = []
        for t, r in radius_samples:
            x = int(plot_x1 + (t / t_max) * plot_w)
            norm = (r - r_min) / (r_max - r_min)
            y = int(plot_y2 - norm * plot_h)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (0, 255, 0), 2)

        # 显示当前半径
        current_r = radii[-1]
        cv2.putText(panel, f"当前 r: {current_r:.2f} px", (plot_x1, plot_y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    text_items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []
    # 显示上一段的平均半径（含中文，使用 PIL 绘制）
    if radius_avg_value is not None:
        text_items.append((f"上一段平均 r: {radius_avg_value:.2f} px", (plot_x1, plot_y1 - 22), 18, (0, 255, 255)))

    # 绘制开始/结束按钮
    btn_color = (0, 200, 0) if not radius_recording else (0, 0, 200)
    cv2.rectangle(panel, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), btn_color, thickness=cv2.FILLED)
    label = "开始记录" if not radius_recording else "结束并计算"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    text_x = BTN_X1 + (BTN_X2 - BTN_X1 - text_size[0]) // 2
    text_y = BTN_Y1 + (BTN_Y2 - BTN_Y1 + text_size[1]) // 2
    # 按钮文字含中文，使用 PIL 绘制
    text_items.append((label, (text_x, text_y - 18), 22, (255, 255, 255)))

    if text_items:
        _draw_text_pil_bgr(panel, text_items)

    cv2.imshow('radius_monitor', panel)

def press_any_key_exit():
    if sys.platform == 'win32':
        return msvcrt.getch()
    else:
        fd = sys.stdin.fileno()
        old_ttyinfo = termios.tcgetattr(fd)
        new_ttyinfo = old_ttyinfo[:]
        new_ttyinfo[3] &= ~termios.ICANON
        new_ttyinfo[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
        try:
            os.read(fd, 7)
        except:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


def IsHBPixelFormat(enPixelType = 0):
    if enPixelType in (PixelType_Gvsp_HB_Mono8, \
                        PixelType_Gvsp_HB_Mono10,\
                        PixelType_Gvsp_HB_Mono10_Packed,\
                        PixelType_Gvsp_HB_Mono12,\
                        PixelType_Gvsp_HB_Mono12_Packed,\
                        PixelType_Gvsp_HB_Mono16,\
                        PixelType_Gvsp_HB_RGB8_Packed,\
                        PixelType_Gvsp_HB_BGR8_Packed,\
                        PixelType_Gvsp_HB_RGBA8_Packed,\
                        PixelType_Gvsp_HB_BGRA8_Packed,\
                        PixelType_Gvsp_HB_RGB16_Packed,\
                        PixelType_Gvsp_HB_BGR16_Packed,\
                        PixelType_Gvsp_HB_RGBA16_Packed,\
                        PixelType_Gvsp_HB_BGRA16_Packed,\
                        PixelType_Gvsp_HB_YUV422_Packed,\
                        PixelType_Gvsp_HB_YUV422_YUYV_Packed,\
                        PixelType_Gvsp_HB_BayerGR8,\
                        PixelType_Gvsp_HB_BayerRG8,\
                        PixelType_Gvsp_HB_BayerGB8,\
                        PixelType_Gvsp_HB_BayerBG8,\
                        PixelType_Gvsp_HB_BayerRBGG8,\
                        PixelType_Gvsp_HB_BayerGB10,\
                        PixelType_Gvsp_HB_BayerGB10_Packed,\
                        PixelType_Gvsp_HB_BayerBG10,\
                        PixelType_Gvsp_HB_BayerBG10_Packed,\
                        PixelType_Gvsp_HB_BayerRG10,\
                        PixelType_Gvsp_HB_BayerRG10_Packed,\
                        PixelType_Gvsp_HB_BayerGR10,\
                        PixelType_Gvsp_HB_BayerGR10_Packed,\
                        PixelType_Gvsp_HB_BayerGB12,\
                        PixelType_Gvsp_HB_BayerGB12_Packed,\
                        PixelType_Gvsp_HB_BayerBG12,\
                        PixelType_Gvsp_HB_BayerBG12_Packed,\
                        PixelType_Gvsp_HB_BayerRG12,\
                        PixelType_Gvsp_HB_BayerRG12_Packed,\
                        PixelType_Gvsp_HB_BayerGR12,\
                        PixelType_Gvsp_HB_BayerGR12_Packed):
        return True
    else:
        return False
    

def IsMonoPixelFormat(enPixelType = 0):
    if enPixelType in (PixelType_Gvsp_Mono8, \
                        PixelType_Gvsp_Mono10, \
                        PixelType_Gvsp_Mono10_Packed, \
                        PixelType_Gvsp_Mono12, \
                        PixelType_Gvsp_Mono12_Packed, \
                        PixelType_Gvsp_Mono14, \
                        PixelType_Gvsp_Mono16):
        return True
    else:
        return False

# 为线程定义一个函数
def work_thread(cam=0, contour_detection=None):
    global g_bExit
    
    stOutFrame = MV_FRAME_OUT()  
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    
    SaveImageOnce = False
    
    # 创建显示窗口
    _named_window('preview', '海康相机 - 实时预览', cv2.WINDOW_NORMAL)
    _named_window('data_panel', '干涉数据面板', cv2.WINDOW_AUTOSIZE)
    # 半径-时间监控窗口
    _named_window('radius_monitor', '半径监控', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('radius_monitor', on_radius_window_mouse)
    print("开始实时显示，按 'q' 键退出...")
    
    while True:
        if g_bExit == True:
            break
            
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            try:
                # 正确的方法：使用ctypes将指针转换为Python可访问的数据
                # 创建一个与缓冲区大小相同的ctypes数组
                buffer_ptr = cast(stOutFrame.pBufAddr, POINTER(c_ubyte * stOutFrame.stFrameInfo.nFrameLen))
                
                # 将ctypes数组转换为numpy数组
                numpy_image = numpy.frombuffer(buffer_ptr.contents, dtype=numpy.uint8)
                
                # 重塑为正确的尺寸
                numpy_image = numpy_image.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth)
                
                # 转换为三通道以便使用干涉条纹检测逻辑
                # color_frame = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2BGR)

                if contour_detection is not None:
                    try:
                        # 直接传入灰度图，fringe_counter 内部会处理
                        display_frame = contour_detection.update(numpy_image)
                    except Exception as err:
                        print(f"干涉条纹处理失败: {err}")
                        display_frame = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2BGR)

                # 显示实时图像
                cv2.imshow('preview', display_frame)

                # 简单数据面板：显示计数值、缩放状态和最近事件
                if contour_detection is not None:
                    panel = numpy.zeros((220, 360, 3), dtype=numpy.uint8)
                    panel[:] = (32, 32, 32)

                    lines: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []
                    # 使用 PIL 绘制中文面板文字
                    lines.append(("Michelson 数据", (10, 8), 22, (0, 255, 0)))
                    lines.append((f"计数: {contour_detection.count_value}", (10, 38), 20, (255, 255, 255)))
                    lines.append((f"状态: {contour_detection.status}", (10, 68), 20, (255, 255, 255)))
                    lines.append((f"环数: {contour_detection.previous_ring_count}", (10, 98), 20, (255, 255, 255)))
                    # put_line(f"外环半径: {contour_detection.previous_outer_radius:.1f}", 4)
                    event = getattr(contour_detection, 'recent_ring_event', None)
                    if event is not None:
                        lines.append((f"事件: {event}", (10, 128), 20, (0, 255, 255)))

                    _draw_text_pil_bgr(panel, lines)

                    cv2.imshow('data_panel', panel)

                # 更新半径-时间监控窗口：使用离中心最近条纹的半径
                if contour_detection is not None:
                    inner_radius = getattr(contour_detection, 'previous_inner_radius', 0.0)
                    if inner_radius is not None and inner_radius > 0:
                        update_radius_window(inner_radius)
                    else:
                        update_radius_window(None)

                # 保存第一张图像
                if not SaveImageOnce:
                    output_path = "output_mono_image.bmp"
                    cv2.imwrite(output_path, numpy_image)
                    print(f"保存图像成功: {output_path}")
                    SaveImageOnce = True
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    g_bExit = True
                elif key == ord('d') and contour_detection is not None:
                    contour_detection.toggle_debug()
                    
            except Exception as e:
                print(f"图像处理错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 释放缓冲区
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            if ret != 0:
                print(f"获取数据失败: 0x{ret:x}")
                
if __name__ == "__main__":

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                  | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0])                
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            print ("\nCML device: [%d]" % i)
            strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            print ("\nCXP device: [%d]" % i)
            strModeName =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)
            
            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
            print ("\nXoF device: [%d]" % i)
            strModeName =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName if c != 0])
            print ("device model name: %s" % strModeName)

            strSerialNumber =  ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber if c != 0])
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = input("please input the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    
    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    
    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
            if ret != 0:
                print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    contour_detection = LineProfileDetection()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, contour_detection))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")
        
    print ("press a key to stop grabbing.")
    press_any_key_exit()

    g_bExit = True
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()