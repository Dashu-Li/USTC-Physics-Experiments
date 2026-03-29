"""Michelson fringe detection helpers derived from https://github.com/julymiaw/physic.

The original repository provides a contour based counter for concentric interference rings.
This module adapts that logic for live CCD streams.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2 as cv
import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, 2]
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def _hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def _wrap_to_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


@dataclass
class LineProfileDetection:
    """Fringe counter based on the intensity profile of a central line.

    Core idea:
    - Extract a 1D intensity profile from the image center (horizontal/vertical line).
    - Estimate dominant spatial frequency via FFT.
    - Project profile onto sin/cos at that frequency to get phase.
    - Unwrap phase over time; every 2π is one fringe passing (count +/- 1).
    """

    settings: Optional[dict] = None

    # External Status Hint
    _stage_moving: bool = False
    _stage_direction: int = 0  # 1 or -1
    _stage_velocity: float = 0.0
    _stage_position: float = 0.0 # Current position in mm
    _external_status_received: bool = False
    
    # Analysis Data
    _reference_wavelength_nm: float = 632.8
    # Fit State
    _fit_data: List[Tuple[float, float]] = field(default_factory=list) # (pos, phase)
    _fit_active_direction: int = 0
    _last_draw_timestamp: float = 0.0
    
    # Manual Configuration
    _manual_roi: Optional[Tuple[int, int, int, int]] = None
    _manual_center: Optional[Tuple[float, float]] = None

    def set_stage_status(self, moving: bool, direction: int = 0, velocity: float = 0.0, position: float = 0.0):
        """Update external stage status to assist counting logic."""
        prev_moving = getattr(self, "_dir_prev_moving", False)
        self._stage_moving = moving
        self._stage_direction = direction
        self._stage_velocity = velocity
        self._stage_position = position
        self._external_status_received = True

        # 推断位移台方向（优先使用传入的 direction，否则用位置增量）
        inferred_dir = direction
        if inferred_dir == 0 and getattr(self, "_dir_last_stage_pos", None) is not None:
            delta_pos = position - self._dir_last_stage_pos
            if delta_pos > 0:
                inferred_dir = 1
            elif delta_pos < 0:
                inferred_dir = -1

        # 记录本次位置，供下次比较
        self._dir_last_stage_pos = position

        # 记录方向变化轨迹（时间, 方向, N1）
        if inferred_dir != 0:
            log = getattr(self, "_dir_motion_log", None)
            if log is not None:
                log.append((time.time(), inferred_dir, self.count_intensity_method))
                if len(log) > 500:
                    del log[:100]

        # 若方向变化（包括停后反向启动），仅在已锁定方向后才同步更新；否则先记录，不抢占 N2 判定
        if inferred_dir != 0 and inferred_dir != getattr(self, "_dir_last_stage_dir", 0):
            # 如果是从停止后重新启动，记录重启方向
            if not prev_moving and moving:
                self._dir_restart_dir = inferred_dir
            # 若已锁定且方向反转，翻转 N1 固定方向
            if getattr(self, "_dir_fixed_initialized", False) and self._latched_direction != 0:
                self._latched_direction = -self._latched_direction
            self._dir_last_stage_dir = inferred_dir
            if getattr(self, "_dir_fixed_initialized", False) and self._latched_direction == 0:
                self._latched_direction = inferred_dir

        # 停止事件：记录最后运动方向
        if prev_moving and not moving:
            self._dir_last_stop_dir = getattr(self, "_dir_last_stage_dir", 0)

        # 停止后再启动：若方向与停前相反，则翻转 N1 固定方向；相同则保持
        if (not prev_moving) and moving and inferred_dir != 0:
            last_stop_dir = getattr(self, "_dir_last_stop_dir", 0)
            if last_stop_dir != 0 and inferred_dir != last_stop_dir:
                if self._latched_direction != 0:
                    self._latched_direction = -self._latched_direction
                    self._dir_fixed_initialized = True
            self._dir_current_move_dir = inferred_dir

        self._dir_prev_moving = moving

    def notify_calibration_done(self):
        """完成标定后调用：开启1秒观测窗口确定固定方向。"""
        self._dir_guard_end_time = time.time() + 1.0
        self._latched_direction = 0
        self._dir_fixed_initialized = False
        self._dir_history = []
        self._dir_n2_ref = self.count_minima_method

    def set_ref_wavelength(self, wave_nm: float):
        self._reference_wavelength_nm = float(wave_nm)

    def set_manual_config(self, roi: Optional[Tuple[int, int, int, int]], center: Optional[Tuple[float, float]]):
        """Set manual ROI (x,y,w,h) and Center (x,y) in global image coordinates."""
        # Check for changes to trigger reset of stabilization only
        if roi != self._manual_roi:
             self._prev_roi_aligned = None
        
        self._manual_roi = roi
        self._manual_center = center

    def reset_fit_data(self):
        """Manually reset the wavelength fitting data."""
        self._fit_data = []
        self._fit_active_direction = 0
        self._last_fit_time = time.time()       

    def set_smoothing_params(self, ms: float):
        """Update smoothing parameter and reset relevant buffers/state to avoid glitches."""
        ms = float(max(0, ms))
        self.settings["temporal_smooth_ms"] = ms
        
        # Clear smoothing buffers so we don't mix old/new time windows
        self._u_profile_buffer = []
        if hasattr(self, "_u_profile_timestamps"):
            self._u_profile_timestamps = []
            
        # Reset tracking state (force re-lock on next frame)
        self._prev_u_profile = None
        self._u_shift_history = []
        
        # Reset analysis/fitting since signal characteristics changed
        self.reset_fit_data()
        
        print(f"[FringeCounter] Smoothing set to {ms}ms. Buffers and Fit reset.")

    def __post_init__(self) -> None:
        defaults = {
            "method": "polar_u",  # 'polar_u' (A+B), fallback: 'line'
            "line_orientation": "h",  # 'h' horizontal center line, 'v' vertical center line
            "line_side": "pos",  # 'pos' from center to + direction, 'neg' from center to - direction, 'both'
            "line_thickness": 5,  # average over this many pixels for robustness
            "roi_fraction": (0.15, 0.85),  # use central portion of the line (start,end) as fraction
            "smooth_sigma": 1.0,  # 1D gaussian smoothing sigma (pixels)
            "min_peak_freq": 4,  # lower to include slower spatial changes
            "max_peak_freq_ratio": 0.45,  # ignore near-Nyquist noise
            "min_signal_strength": 0.02,  # more permissive for weak fringes
            "count_per_fringe": 1,  # count increment per 1 fringe
            "invert_direction": True,  # flip swallow/spit sign (default for this setup)
            "min_corr_peak": 0.08,  # lower correlation peak threshold for sensitivity
            "max_abs_shift_px": 15.0,  # allow larger per-frame shift
            "min_abs_delta_fringe": 0.03,  # smaller deadband to catch micro moves
            "direction_confirm_frames": 1,  # accept single-frame consistent direction
            "shift_median_window": 3,  # shorter median window for responsiveness
            "draw_line": True,
            "draw_bbox": True,
            "draw_profile": True,
            "segment_kernel": (5, 5),
            "segment_kernel_fallback": (7, 7),
            "max_bbox_area_jump": 1.5,
            "stabilize_roi": True,
            "stabilize_min_response": 0.04,
            "center_blur_sigma": 5.0,
            "polar_angle_samples": 1080,
            "polar_radius_fraction": 0.98,
            "angular_stat": "median",  # 'mean' or 'median'
            "u_samples": 512,
            "radial_inner_min_px": 10,
            "temporal_smooth_ms": 0, # Averaging time window in milliseconds (0=off)
            "sat_time_thresh_ms": 180, # Saturation duration threshold in ms
        }
        self.settings = {**defaults, **(self.settings or {})}
        self.reset_counters()
        self.reset_state()
        
        # New independent counter for I(0) minimum method
        self.count_intensity_method = 0 # N1: Saturation count
        self.count_minima_method = 0    # N2: Minima count
        self._last_intensity_val = 0.0
        self._intensity_trend = 0 # 0: calculating, 1: rising, -1: falling
        self._in_valley_state = False # State for N2

        # Reporting state for N1 per 10 counts
        self._n1_checkpoint_val = 0
        self._n1_checkpoint_pos = 0.0

    def reset_counters(self) -> None:
        self.count_value = 0
        self.total_frames = 0
        self.count_intensity_method = 0 # Reset N1
        self.count_minima_method = 0    # Reset N2
        self._in_valley_state = False
        self._n1_checkpoint_val = 0
        self._n1_checkpoint_pos = self._stage_position if hasattr(self, "_stage_position") else 0.0
        self._latched_direction = 0 # N1固定方向（单次判定）
        self._dir_fixed_initialized = False
        self._dir_guard_end_time = 0.0
        self._dir_history: List[Tuple[float, float]] = []
        self._dir_last_stage_dir = 0
        self._dir_last_stage_pos: Optional[float] = None
        self._dir_n2_ref: Optional[float] = None
        self._dir_last_stop_dir = 0
        self._dir_prev_moving = False
        self._dir_current_move_dir = 0
        self._dir_motion_log: List[Tuple[float, int, float]] = []
        self._dir_restart_dir = 0

    def reset_state(self) -> None:
        self.status: Optional[str] = None
        self.recent_ring_event: Optional[str] = None
        self.recent_ring_event_timer = 0
        self.last_output_frame: Optional[np.ndarray] = None
        self.previous_inner_radius = 0.0
        self.previous_ring_count = 0
        self._prev_phase: Optional[float] = None
        self._phase_accum = 0.0
        self._k_peak: Optional[int] = None
        self._last_strength: float = 0.0
        self._prev_profile: Optional[np.ndarray] = None
        self._shift_accum = 0.0
        self._bbox_origin: Tuple[int, int] = (0, 0)
        self._bbox: BBox = (0, 0, 0, 0)
        self._previous_bbox_area: float = 0.1
        self._direction_sign: int = 0
        self._direction_persist: int = 0
        self._shift_history: List[float] = []
        self._analysis_history: List[Tuple[float, float]] = [] # Reset analysis data too
        self._prev_roi_aligned: Optional[np.ndarray] = None
        self._center_xy: Optional[Tuple[float, float]] = None

        self._prev_u_profile: Optional[np.ndarray] = None
        self._u_shift_accum = 0.0
        self._u_shift_history: List[float] = []
        
        # New history buffer for temporal smoothing of the raw u_profile
        self._u_profile_buffer: List[np.ndarray] = []
        self._u_profile_timestamps: List[float] = [] # timestamps for time-based smoothing

        # History for Phase Plotting (Total Fringes vs Time)
        self._phase_plot_history: List[float] = [] 
        
        # Fit logic reset
        self._fit_data = []
        self._fit_active_direction = 0
        self._last_draw_timestamp = 0.0

    def toggle_debug(self) -> None:
        current = bool(self.settings.get("draw_profile", False))
        self.settings["draw_profile"] = not current
        mode = "剖面" if self.settings["draw_profile"] else "关闭剖面"
        print(f"[LineProfileDetection] 已切换: {mode}")

    def _extract_profile(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        orientation = str(self.settings.get("line_orientation", "h")).lower()
        side = str(self.settings.get("line_side", "pos")).lower()
        thickness = int(max(1, self.settings.get("line_thickness", 5)))
        thickness = min(thickness, h if orientation == "h" else w)
        half = thickness // 2

        if orientation == "v":
            x0 = w // 2
            x1 = max(0, x0 - half)
            x2 = min(w, x0 + half + 1)
            stripe = gray[:, x1:x2]
            profile_full = stripe.mean(axis=1)
        else:
            y0 = h // 2
            y1 = max(0, y0 - half)
            y2 = min(h, y0 + half + 1)
            stripe = gray[y1:y2, :]
            profile_full = stripe.mean(axis=0)

        profile_full = profile_full.astype(np.float32)

        # Use half-line to avoid symmetric cancellation for concentric ring expansion.
        if side in ("pos", "positive", "+"):
            c = profile_full.size // 2
            profile = profile_full[c:]
        elif side in ("neg", "negative", "-"):
            c = profile_full.size // 2
            # flip so that increasing index still means moving away from center
            profile = profile_full[:c][::-1]
        else:
            profile = profile_full

        start_f, end_f = self.settings.get("roi_fraction", (0.15, 0.85))
        start = int(np.clip(float(start_f), 0.0, 1.0) * (profile.size - 1))
        end = int(np.clip(float(end_f), 0.0, 1.0) * (profile.size - 1))
        if end <= start + 8:
            return profile
        return profile[start:end]

    def _localize_and_crop(self, gray: np.ndarray) -> Tuple[np.ndarray, BBox, Tuple[int, int]]:
        """Reuse the existing segmentation logic to localize the fringe region.

        Returns (cropped_gray, bbox, bbox_origin).
        """
        h, w = gray.shape[:2]

        # 1. Manual ROI Override
        if self._manual_roi is not None:
            mx, my, mw, mh = self._manual_roi
            # Clamp to image bounds
            mx = max(0, min(mx, w - 1))
            my = max(0, min(my, h - 1))
            mw = max(1, min(mw, w - mx))
            mh = max(1, min(mh, h - my))
            
            bbox = (mx, my, mw, mh)
            cropped = gray[my : my + mh, mx : mx + mw].copy()
            return cropped, bbox, (mx, my)

        # FIXED CENTER SQUARE ROI (Half Width)
        
        # Side length is half of the image width
        side = int(w // 2)
        
        # Calculate top-left corner to center the square
        x = int(w // 2 - side // 2)
        y = int(h // 2 - side // 2)
        
        # If height is too small, clamp y to 0 and reduce side if necessary to stay square
        if y < 0:
            y = 0
            if h < side:
                side = h
                x = int(w // 2 - side // 2)
                
        # Basic bounds checking
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        side = max(1, min(side, w - x, h - y))
        
        bbox = (x, y, side, side)
        cropped = gray[y : y + side, x : x + side]
        
        return cropped, bbox, (x, y)

    def _smooth_1d(self, x: np.ndarray) -> np.ndarray:
        sigma = float(self.settings.get("smooth_sigma", 1.5))
        if sigma <= 0.01 or x.size < 5:
            return x
        # Use OpenCV GaussianBlur on a 1xN image for speed/robustness.
        x2 = x.reshape(1, -1)
        # ksize=(0,0) lets OpenCV derive kernel from sigma.
        blurred = cv.GaussianBlur(x2, (0, 0), sigmaX=sigma, sigmaY=0)
        return blurred.reshape(-1)

    def _phase_correlate_2d(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[Tuple[float, float], float]:
        """Estimate translation that aligns cur to ref using 2D phase correlation."""
        ref32 = ref.astype(np.float32)
        cur32 = cur.astype(np.float32)
        # windowing improves robustness against borders
        win = cv.createHanningWindow((ref32.shape[1], ref32.shape[0]), cv.CV_32F)
        try:
            shift, response = cv.phaseCorrelate(ref32, cur32, win)
        except Exception:
            shift, response = cv.phaseCorrelate(ref32, cur32)
        return (float(shift[0]), float(shift[1])), float(response)

    def _warp_translate(self, img: np.ndarray, dx: float, dy: float) -> np.ndarray:
        h, w = img.shape[:2]
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv.warpAffine(img, m, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    def _estimate_center_peak(self, gray: np.ndarray) -> Tuple[float, float]:
        """Estimate fringe center using Hough Circles with fallback to intensity peak + smoothing."""
        h, w = gray.shape[:2]
        return float(w) / 2.0, float(h) / 2.0

        # --- Strategy 1: Hough Circles (Geometric Center) ---
        # Robust against intensity changes (bright/dark center)
        # Use a blurred copy to reduce noise constraints
        blurred = cv.GaussianBlur(gray, (9, 9), 2)
        
        # Param1: Canny high threshold (lower = more edges)
        # Param2: Accumulator threshold (lower = more circles, higher = strict)
        # minDist: Min distance between centers. Since we want CONCENTRIC circles, 
        # we expect them to be close. We can set it small and average them.
        circles = cv.HoughCircles(
            blurred, 
            cv.HOUGH_GRADIENT, 
            dp=1, 
            minDist=w/20,  # Allow multiple concentric circles
            param1=100, 
            param2=30, 
            minRadius=20, 
            maxRadius=min(h, w)//2
        )
        
        meas_cx, meas_cy = None, None

        if circles is not None:
            # circles: [[[x, y, r], ...]]
            circles = np.uint16(np.around(circles))
            centers = circles[0, :, :2]
            # Use median to reject outliers
            median_center = np.median(centers, axis=0)
            meas_cx, meas_cy = float(median_center[0]), float(median_center[1])

        # --- Strategy 2: Peak Intensity (Fallback) ---
        if meas_cx is None:
            sigma = float(self.settings.get("center_blur_sigma", 5.0))
            if sigma > 0.01:
                blurred_peak = cv.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
            else:
                blurred_peak = gray
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(blurred_peak)
            
            # Sanity check: Peak must be somewhat central (inner 60%) to be trusted as center
            # otherwise it might be a reflection at the edge
            px, py = float(max_loc[0]), float(max_loc[1])
            if 0.2 * w < px < 0.8 * w and 0.2 * h < py < 0.8 * h:
                 meas_cx, meas_cy = px, py
            else:
                 # Last resort: Simply image center
                 meas_cx, meas_cy = w / 2.0, h / 2.0

        # --- Temporal Smoothing ---
        # If we have no history, take the measurement
        if self._center_xy is None:
            return meas_cx, meas_cy
        
        old_x, old_y = self._center_xy
        
        # Calculate distance from old center
        dist = math.hypot(meas_cx - old_x, meas_cy - old_y)
        
        # Adaptive smoothing factor alpha
        # If change is small, smooth heavily to reduce jitter (small alpha)
        # If change is large (but not impossible), assume camera moved, respond faster (larger alpha)
        # If change is HUGE, it might be a glitch, ignore or damp heavily?
        # Here we just use a simple Low Pass Filter.
        
        # Hough is usually accurate but can jitter.
        # If Hough found something, we trust it more.
        if circles is not None:
            alpha = 0.2  # 20% update rate for geometric
        else:
            alpha = 0.05 # 5% update rate for peak (very unstable during swallow/spit)

        # If the shift is massive (> 1/4 image), reset immediately (camera moved setup)
        if dist > min(w, h) / 4.0:
            alpha = 1.0
            
        new_x = old_x * (1.0 - alpha) + meas_cx * alpha
        new_y = old_y * (1.0 - alpha) + meas_cy * alpha
        
        return new_x, new_y

    def _radial_profile_polar_u(self, gray: np.ndarray, center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        """Return (radial_profile_r, u_profile, max_radius_used, fft_mag)."""
        h, w = gray.shape[:2]
        cx, cy = center_xy
        # maximum radius constrained by image borders
        max_r_border = min(cx, cy, float(w - 1) - cx, float(h - 1) - cy)
        max_r_border = max(1.0, max_r_border)
        max_r = float(self.settings.get("polar_radius_fraction", 0.98)) * max_r_border
        max_r = max(16.0, max_r)
        max_r_i = int(max_r)

        angle_samples = int(max(180, self.settings.get("polar_angle_samples", 720)))
        # warpPolar output: (height=angles, width=radius)
        polar = cv.warpPolar(
            gray,
            (max_r_i, angle_samples),
            (cx, cy),
            max_r,
            flags=cv.WARP_POLAR_LINEAR,
        )
        polar = polar.astype(np.float32)

        stat = str(self.settings.get("angular_stat", "median")).lower()
        if stat == "mean":
            radial_r = polar.mean(axis=0)
        else:
            radial_r = np.median(polar, axis=0)

        radial_r = radial_r.astype(np.float32)
        # ignore very small radii (often saturated/flat)
        r0 = int(max(0, self.settings.get("radial_inner_min_px", 10)))
        if radial_r.size > r0 + 8:
            radial_r = radial_r[r0:]

        # resample from r to u=r^2 with uniform u spacing
        r = np.arange(radial_r.size, dtype=np.float32)
        u = r * r
        u_max = float(u[-1]) if u.size else 1.0
        n_u = int(max(128, self.settings.get("u_samples", 512)))
        u_lin = np.linspace(0.0, u_max, num=n_u, dtype=np.float32)
        r_from_u = np.sqrt(u_lin)
        u_profile = np.interp(r_from_u, r, radial_r).astype(np.float32)
        
        # Calculate FFT of u_profile for visualization
        mag = None
        if u_profile.size > 32:
             # Remove DC component
             u_no_dc = u_profile - np.mean(u_profile) 
             # Apply window
             window = _hann_window(u_no_dc.size)
             u_windowed = u_no_dc * window
             # Compute FFT
             fft_spec = np.fft.rfft(u_windowed)
             mag = np.abs(fft_spec).astype(np.float32)
             mag[0] = 0.0 # Clear DC just in case

        return radial_r, u_profile, max_r, mag

    def _estimate_inner_radius(self, radial_r: np.ndarray) -> float:
        """Rough estimate of the first dark ring radius (for compatibility with the radius monitor)."""
        if radial_r.size < 32:
            return 0.0
        x = self._smooth_1d(radial_r)
        # look for first valley after a small offset
        start = 12
        end = min(int(radial_r.size * 0.6), radial_r.size - 1)
        if end <= start + 5:
            return 0.0
        idx = int(start + int(np.argmin(x[start:end])))
        return float(idx)

    def _estimate_k_peak(self, signal: np.ndarray) -> Optional[int]:
        n = int(signal.size)
        if n < 32:
            return None
        window = _hann_window(n)
        s = (signal - float(np.mean(signal))) * window
        # rfft gives bins 0..n//2
        spec = np.fft.rfft(s)
        mag = np.abs(spec).astype(np.float32)
        mag[0] = 0.0

        k_min = int(max(1, self.settings.get("min_peak_freq", 3)))
        k_max = int(min(len(mag) - 1, max(2, int(float(self.settings.get("max_peak_freq_ratio", 0.45)) * n))))
        if k_max <= k_min:
            return None
        k = int(k_min + int(np.argmax(mag[k_min : k_max + 1])))
        return k

    def _phase_from_projection(self, signal: np.ndarray, k: int) -> Tuple[float, float]:
        n = int(signal.size)
        if n <= 0:
            return 0.0, 0.0
        x = np.arange(n, dtype=np.float32)
        w = (2.0 * math.pi * float(k)) / float(n)
        cos_t = np.cos(w * x)
        sin_t = np.sin(w * x)
        s = signal - float(np.mean(signal))
        denom = float(np.linalg.norm(s) + 1e-6)
        c = float(np.dot(s, cos_t) / denom)
        si = float(np.dot(s, sin_t) / denom)
        strength = float(math.sqrt(c * c + si * si))
        phase = float(math.atan2(si, c))
        return phase, strength

    def _phase_correlation_shift(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, Optional[np.ndarray]]:
        """Estimate 1D shift between a (prev) and b (current) using phase correlation.

        Returns (shift_pixels, peak_quality, correlation_array).
        Positive shift means b is shifted to the right relative to a.
        """
        n = int(min(a.size, b.size))
        if n < 32:
            return 0.0, 0.0, None
        a0 = a[:n].astype(np.float32)
        b0 = b[:n].astype(np.float32)
        a0 = a0 - float(np.mean(a0))
        b0 = b0 - float(np.mean(b0))
        win = _hann_window(n)
        a0 *= win
        b0 *= win
        A = np.fft.rfft(a0)
        B = np.fft.rfft(b0)
        R = A * np.conj(B)
        denom = np.abs(R)
        R /= (denom + 1e-9)
        corr = np.fft.irfft(R, n=n)
        k = int(np.argmax(corr))
        peak = float(corr[k])
        # convert peak index to signed lag in [-n/2, n/2)
        if k > n // 2:
            lag = k - n
        else:
            lag = k
        return float(lag), peak, corr

    def _update_event_timer(self, ring_event: Optional[str]) -> None:
        if ring_event is not None:
            self.recent_ring_event = ring_event
            self.recent_ring_event_timer = 30
        elif self.recent_ring_event_timer > 0:
            self.recent_ring_event_timer -= 1
            if self.recent_ring_event_timer == 0:
                self.recent_ring_event = None

    def update(self, frame: np.ndarray) -> np.ndarray:
        self.total_frames += 1

        # 记录用于方向判定的历史 N-时间数据（保留最近3秒）
        now_t = time.time()
        total_phase_val_for_dir = float(self.count_value) + getattr(self, "_u_shift_accum", 0.0)
        self._dir_history.append((now_t, total_phase_val_for_dir))
        cutoff = now_t - 3.0
        while len(self._dir_history) > 0 and self._dir_history[0][0] < cutoff:
            self._dir_history.pop(0)

        gray_full = _to_gray(frame)
        gray_roi, bbox, bbox_origin = self._localize_and_crop(gray_full)
        self._bbox = bbox
        self._bbox_origin = bbox_origin

        method = str(self.settings.get("method", "polar_u")).lower()

        ring_event: Optional[str] = None
        delta_fringe_dbg = 0.0
        peak_q_dbg = 0.0
        period_px_dbg = 0.0
        stabilize_dbg = 0.0
        center_dbg: Optional[Tuple[float, float]] = None

        # --- A) Stabilize ROI against camera jitter ---
        roi_aligned = gray_roi
        dx_val, dy_val = 0.0, 0.0
        if bool(self.settings.get("stabilize_roi", True)):
            if self._prev_roi_aligned is None or self._prev_roi_aligned.shape != gray_roi.shape:
                self._prev_roi_aligned = gray_roi.astype(np.float32)
            else:
                (dx, dy), resp = self._phase_correlate_2d(self._prev_roi_aligned, gray_roi)
                stabilize_dbg = resp
                if resp >= float(self.settings.get("stabilize_min_response", 0.05)):
                    roi_aligned = self._warp_translate(gray_roi, dx, dy)
                    dx_val, dy_val = dx, dy
                self._prev_roi_aligned = roi_aligned.astype(np.float32)

        # --- B) Polar unwrap + angular aggregation + u=r^2 resampling ---
        if method.startswith("polar"):
            # cx, cy = self._estimate_center_peak(roi_aligned)
            
            h_full, w_full = gray_full.shape[:2]
            ox, oy = bbox_origin

            cx: float = 0.0
            cy: float = 0.0

            if self._manual_center is not None:
                # Manual Center (Global Coords) -> ROI Coords
                mcx, mcy = self._manual_center
                cx_roi = mcx - float(ox)
                cy_roi = mcy - float(oy)
                
                # Apply stabilization offset if active
                cx = cx_roi + dx_val
                cy = cy_roi + dy_val
            else:
                # --- FORCE CENTER TO CAMERA CENTER (PHYSICAL OPTICAL AXIS) ---
                # Calculate the center of the full frame relative to the current ROI
                # The physical center in the original unshifted ROI
                cx_roi = (float(w_full) / 2.0) - float(ox)
                cy_roi = (float(h_full) / 2.0) - float(oy)
                
                # Compensation for stabilization shift:
                cx = cx_roi + dx_val
                cy = cy_roi + dy_val
            
            self._center_xy = (cx, cy)
            center_dbg = (cx, cy)
            # Use 4-element unpacking but ignore mag initially to fix scope issues, we'll assign it to a debug var later
            radial_r, u_profile, _, mag_debug = self._radial_profile_polar_u(roi_aligned, (cx, cy))
            self.previous_inner_radius = self._estimate_inner_radius(radial_r)

            # --- Temporal Smoothing for u_profile (Anti-Jitter) ---
            t_ms = float(self.settings.get("temporal_smooth_ms", 0))
            t_win = int(self.settings.get("temporal_smooth_frames", 1))

            if t_ms > 0:
                now_t = time.time()
                # Ensure timestamps list exists and is synced
                if not hasattr(self, "_u_profile_timestamps"):
                    self._u_profile_timestamps = []
                if len(self._u_profile_buffer) != len(self._u_profile_timestamps):
                    self._u_profile_buffer = []  # Reset on sync loss or mode switch
                    self._u_profile_timestamps = []

                self._u_profile_buffer.append(u_profile)
                self._u_profile_timestamps.append(now_t)

                # Prune frames older than t_ms window
                limit_t = now_t - (t_ms / 1000.0)
                while len(self._u_profile_timestamps) > 0 and self._u_profile_timestamps[0] < limit_t:
                    self._u_profile_timestamps.pop(0)
                    self._u_profile_buffer.pop(0)
                
                if len(self._u_profile_buffer) > 0:
                    u_profile = np.mean(np.array(self._u_profile_buffer), axis=0).astype(np.float32)

            elif t_win > 1:
                # Mode clean up (if switching from MS to Frames)
                if hasattr(self, "_u_profile_timestamps") and len(self._u_profile_timestamps) > 0:
                     self._u_profile_timestamps = []
                     while len(self._u_profile_buffer) >= t_win:
                         self._u_profile_buffer.pop(0)

                self._u_profile_buffer.append(u_profile)
                if len(self._u_profile_buffer) > t_win:
                    self._u_profile_buffer.pop(0)
                
                # Compute average if we have buffer
                if len(self._u_profile_buffer) > 0:
                    u_profile = np.mean(np.array(self._u_profile_buffer), axis=0).astype(np.float32)
            else:
                 # Clear buffer if smoothing is disabled to save memory/state
                 if len(self._u_profile_buffer) > 0:
                      self._u_profile_buffer = []
                      if hasattr(self, "_u_profile_timestamps"):
                          self._u_profile_timestamps = []

            u_profile = self._smooth_1d(u_profile)
            k_new = self._estimate_k_peak(u_profile)
            if k_new is not None:
                self._k_peak = k_new

            strength = 0.0
            if self._k_peak is not None:
                _, strength = self._phase_from_projection(u_profile, self._k_peak)
            self._last_strength = float(strength)
            
            # --- Expose Visibility Metric for Sodium Experiment ---
            # Visibility is roughly proportional to the FFT peak strength or correlation response
            self.current_visibility = self._last_strength
            
            min_strength = float(self.settings.get("min_signal_strength", 0.04))

            if self._prev_u_profile is None or self._prev_u_profile.shape != u_profile.shape:
                self._prev_u_profile = u_profile.copy()
                self._u_shift_history = []
                self._u_shift_accum = 0.0
                self._direction_sign = 0
                self._direction_persist = 0
                self.status = "idle"
            elif self._k_peak is None or self._last_strength < min_strength:
                self._prev_u_profile = u_profile.copy()
                self._u_shift_history = []
                self._direction_sign = 0
                self._direction_persist = 0
                self.status = "idle"
            else:
                shift_samp, peak_q, corr_debug = self._phase_correlation_shift(self._prev_u_profile, u_profile)
                self._prev_u_profile = u_profile.copy()
                peak_q_dbg = float(peak_q)

                # Gating by correlation peak
                if float(peak_q) < float(self.settings.get("min_corr_peak", 0.16)):
                    self.status = "idle"
                    self._u_shift_history = []
                    self._direction_sign = 0
                    self._direction_persist = 0
                else:
                    max_shift = float(self.settings.get("max_abs_shift_px", 10.0))
                    if max_shift > 0:
                        shift_samp = float(np.clip(shift_samp, -max_shift, max_shift))

                    # Median filter recent shifts (still helpful after stabilization)
                    win = int(self.settings.get("shift_median_window", 5))
                    if win < 1:
                        win = 1
                    if win % 2 == 0:
                        win += 1
                    self._u_shift_history.append(float(shift_samp))
                    if len(self._u_shift_history) > win:
                        self._u_shift_history = self._u_shift_history[-win:]
                    shift_samp = float(np.median(np.array(self._u_shift_history, dtype=np.float32)))

                    n = int(u_profile.size)
                    period_samp = (float(n) / float(self._k_peak)) if (self._k_peak and self._k_peak > 0) else 0.0
                    period_px_dbg = float(period_samp)
                    if period_samp > 1e-3:
                        delta_fringe = float(shift_samp / period_samp)
                        # "吐为正，吞为负" uses invert_direction to match your setup.
                        if bool(self.settings.get("invert_direction", True)):
                            delta_fringe = -delta_fringe
                        delta_fringe_dbg = float(delta_fringe)

                        deadband = float(self.settings.get("min_abs_delta_fringe", 0.06))
                        
                        # Dynamic adjustment based on external stage status
                        if getattr(self, "_external_status_received", False):
                            if self._stage_moving:
                                # When moving, force high sensitivity to catch slow fringes
                                deadband = min(deadband, 0.005)
                            else:
                                # When stopped, increase deadband to suppress jitter
                                deadband = max(deadband, 0.03)

                        if abs(delta_fringe) < deadband:
                            self.status = "idle"
                            self._direction_sign = 0
                            self._direction_persist = 0
                        else:
                            sign = 1 if delta_fringe > 0 else -1
                            if sign != self._direction_sign:
                                self._direction_sign = sign
                                self._direction_persist = 1
                            else:
                                self._direction_persist += 1

                            confirm = int(max(1, self.settings.get("direction_confirm_frames", 2)))
                            if self._direction_persist >= confirm:
                                self._u_shift_accum += delta_fringe
                                step = int(max(1, self.settings.get("count_per_fringe", 1)))

                                while self._u_shift_accum >= 1.0:
                                    self.count_value += step
                                    self._u_shift_accum -= 1.0
                                    ring_event = "spit"
                                while self._u_shift_accum <= -1.0:
                                    self.count_value -= step
                                    self._u_shift_accum += 1.0
                                    ring_event = "swallow"

                            self.status = "zoom_out" if sign > 0 else "zoom_in"
                    else:
                        self.status = "idle"
            
            # Make the u_profile available for drawing
            profile = u_profile
            
            # --- Independent Intensity Minima Counting (Validation Method) ---
            # Logic: Check I(u=0) over time. If it stays saturated (Max Brightness) for long enough, count it.
            # This detects the "Inverted Basin" plateau passing the center.
            if u_profile.size > 5:
                # Average center 3 pixels
                curr_intensity = np.mean(u_profile[:3])

                # --- Saturation Duration Counting ---
                SAT_THRESH = 249.0 # Threshold for "Top of the basin"
                # SAT_TIME_THRESH = 0.02 # Old hardcoded
                sat_ms = float(self.settings.get("sat_time_thresh_ms", 180.0))
                SAT_TIME_THRESH = sat_ms / 1000.0

                if curr_intensity >= SAT_THRESH:
                    if not hasattr(self, "_sat_start_time") or self._sat_start_time is None:
                        self._sat_start_time = time.time()
                    self._current_sat_duration = time.time() - self._sat_start_time
                else:
                    self._current_sat_duration = 0.0
                    if hasattr(self, "_sat_start_time") and self._sat_start_time is not None:
                        duration = time.time() - self._sat_start_time
                        
                        if duration > SAT_TIME_THRESH:
                             # Valid Bright Fringe Passage (Plateau ended)
                             direction_vote = 1
                             if self._direction_sign != 0:
                                 direction_vote = self._direction_sign
                             elif self._u_shift_accum > 0:
                                 direction_vote = 1
                             elif self._u_shift_accum < 0:
                                 direction_vote = -1
                            
                             # --- N1 固定方向判定与去抖 ---
                             should_count = True
                             now_t = time.time()

                             # 方向确定策略：
                             # 1) 完成标定后等待 1s 观测窗口，不计数。
                             # 2) 用 FFT 方法得到的 N-t 斜率判定吐/吞，一旦确定固定方向，后续沿此方向计数。
                             # 3) 若位移台方向在行进中改变，set_stage_status 已同步更新固定方向。

                             # 如果处于观测窗口内，暂不计数
                             if now_t < getattr(self, "_dir_guard_end_time", 0):
                                 should_count = False

                             # 若尚未确定固定方向，则用 N2 的变化方向（在标定后1秒内的净增减）确定
                             if should_count and not self._dir_fixed_initialized:
                                 dir_sign = 0
                                 if self._dir_n2_ref is not None:
                                     delta_n2 = self.count_minima_method - self._dir_n2_ref
                                     if delta_n2 > 0:
                                         dir_sign = 1
                                     elif delta_n2 < 0:
                                         dir_sign = -1
                                 if dir_sign != 0:
                                     self._latched_direction = dir_sign
                                     self._dir_fixed_initialized = True

                             # 应用固定方向（若已确定）；否则用当前事件的方向并锁定
                             if self._latched_direction != 0:
                                 direction_vote = self._latched_direction
                             else:
                                 self._latched_direction = direction_vote if direction_vote != 0 else 0
                                 if self._latched_direction != 0:
                                     self._dir_fixed_initialized = True

                             # 时间去抖：1.5x饱和时长
                             min_interval = 1.5 * SAT_TIME_THRESH
                             if hasattr(self, "_last_n1_event") and self._last_n1_event is not None:
                                 last_t, _, _ = self._last_n1_event
                                 if (now_t - last_t) < min_interval:
                                     should_count = False

                             if should_count and direction_vote != 0:
                                 self.count_intensity_method += direction_vote
                                 self._last_n1_event = (now_t, direction_vote, duration)
                                 # print(f"[Intensity Method] Counted N1 (Sat Duration: {duration * 1000.0:.1f} ms)")

                                 # --- N1 Interval Reporting (Every 10 counts) ---
                                 # Calculate Ratio = (Delta_Pos * 2) / Delta_N
                                 delta_n = abs(self.count_intensity_method - self._n1_checkpoint_val)
                                 if delta_n >= 10:
                                     current_pos = self._stage_position
                                     delta_pos = current_pos - self._n1_checkpoint_pos
                                     
                                     # Avoid division by zero, though delta_n >= 10 guarantees it
                                     ratio = (delta_pos * 2.0) / float(delta_n)
                                     # Convert to nm for readability
                                     ratio_nm = ratio * 1e6
                                     
                                     report_msg = f"\n[N1 REPORT] Delta N={delta_n} | Delta Pos={delta_pos*1000:.4f} um | Ratio (Lambda) = {ratio_nm:.4f} nm\n"
                                     print(report_msg)
                                     
                                     # Update checkpoints
                                     self._n1_checkpoint_val = self.count_intensity_method
                                     self._n1_checkpoint_pos = current_pos
                        
                        self._sat_start_time = None
                
                # --- N2: Independent Local Minima Detection ---
                # Detect inflection point (Falling -> Rising)
                delta = curr_intensity - self._last_intensity_val
                NOISE_GATE = 2.0 # Ignore small jitter

                if delta > NOISE_GATE: # Rising
                     if self._intensity_trend == -1: 
                         # Turnaround: Falling -> Rising (Minimum)
                         # Filter: Don't count minima that are actually just noise on top of saturation (e.g. 253->250->254)
                         if self._last_intensity_val < 220.0:
                             d_vote = 1
                             if self._direction_sign != 0: d_vote = self._direction_sign
                             elif self._u_shift_accum > 0: d_vote = 1
                             elif self._u_shift_accum < 0: d_vote = -1

                             self.count_minima_method += d_vote
                             # print(f"[Minima Method] Counted N2 (Local Min {self._last_intensity_val:.1f})")
                     
                     self._intensity_trend = 1
                elif delta < -NOISE_GATE: # Falling
                     self._intensity_trend = -1

                self._last_intensity_val = curr_intensity

        else:
            # Fallback: keep the previous simple line-based logic.
            profile = self._extract_profile(roi_aligned)
            profile = self._smooth_1d(profile)
            k_new = self._estimate_k_peak(profile)
            if k_new is not None:
                self._k_peak = k_new
            self.status = "idle"
        self._update_event_timer(ring_event)

        # Prepare output canvas
        if frame.ndim == 2:
            output = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            output = frame.copy()

        # Border color by event
        if ring_event == "spit":
            border_color = (0, 255, 0)
        elif ring_event == "swallow":
            border_color = (255, 0, 0)
        else:
            border_color = (255, 255, 255)

        border_thickness = 12
        output = cv.copyMakeBorder(
            output,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv.BORDER_CONSTANT,
            value=border_color,
        )

        # Draw bbox and central line inside the localized ROI (in bordered coordinates)
        if bool(self.settings.get("draw_bbox", True)):
            x, y, w, h = bbox
            x1 = int(x + border_thickness)
            y1 = int(y + border_thickness)
            x2 = int(x + w + border_thickness)
            y2 = int(y + h + border_thickness)
            cv.rectangle(output, (x1, y1), (x2, y2), (0, 200, 200), 1)

        # Draw estimated center point (within bbox region)
        if center_dbg is not None:
            cx, cy = center_dbg
            # cx, cy are in bbox-relative coordinates (or if using full image, image relative)
            # but bbox[0], bbox[1] are added to map back to original frame
            # then border_thickness is added for display
            cv.circle(
                output,
                (int(round(cx + bbox[0] + border_thickness)), int(round(cy + bbox[1] + border_thickness))),
                3,
                (0, 0, 255),
                -1,
            )

        if bool(self.settings.get("draw_line", True)):
            orientation = str(self.settings.get("line_orientation", "h")).lower()
            x, y, w, h = bbox
            if orientation == "v":
                x_line = int(x + w // 2 + border_thickness)
                cv.line(
                    output,
                    (x_line, int(y + border_thickness)),
                    (x_line, int(y + h + border_thickness)),
                    (0, 255, 255),
                    1,
                    cv.LINE_AA,
                )
            else:
                y_line = int(y + h // 2 + border_thickness)
                cv.line(
                    output,
                    (int(x + border_thickness), y_line),
                    (int(x + w + border_thickness), y_line),
                    (0, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        # Main count text
        h_canvas, w_canvas = output.shape[:2]
        font_scale = max(1.2, min(w_canvas, h_canvas) / 180.0)
        thickness = max(2, int(font_scale * 2.5))
        
        # Display N1 (Intensity Method) as Main Count
        count_text = f"{self.count_intensity_method}"
        
        (text_w, text_h), _ = cv.getTextSize(count_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = w_canvas - text_w - 20
        text_y = text_h + 20
        cv.putText(output, count_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4, cv.LINE_AA)
        cv.putText(output, count_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv.LINE_AA)

        # Draw Original N Counter at Bottom-Right
        total_phase_val = float(self.count_value) + self._u_shift_accum
        n1_text = f"N: {total_phase_val:.3f}"
        # Use slightly smaller font for N1
        n1_font_scale = font_scale * 0.7
        n1_thickness = max(2, int(thickness * 0.7))
        (n1_w, n1_h), _ = cv.getTextSize(n1_text, cv.FONT_HERSHEY_SIMPLEX, n1_font_scale, n1_thickness)
        
        n1_x = w_canvas - n1_w - 20
        n1_y = h_canvas - 20
        
        cv.putText(output, n1_text, (n1_x, n1_y), cv.FONT_HERSHEY_SIMPLEX, n1_font_scale, (0, 0, 0), n1_thickness + 3, cv.LINE_AA)
        # Use Orange/Gold to distinguish N1
        cv.putText(output, n1_text, (n1_x, n1_y), cv.FONT_HERSHEY_SIMPLEX, n1_font_scale, (0, 165, 255), n1_thickness, cv.LINE_AA)

        # Optional profile debug window
        if bool(self.settings.get("draw_profile", False)):
            # Create a separate window for detailed analysis
            H_win, W_win = 600, 800
            canvas = np.zeros((H_win, W_win, 3), dtype=np.uint8)
            h_plot = H_win // 3
            
            # --- 1. Profile (Top) ---
            cv.rectangle(canvas, (0, 0), (W_win, h_plot), (30, 30, 30), -1)
            cv.putText(canvas, f"Space Domain Profile (u=r^2)", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            msg_i = f"N1(Sat):{self.count_intensity_method}  N2(Min):{self.count_minima_method}"
            cv.putText(canvas, msg_i, (W_win - 350, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            # --- N1 Visualization (Real-time Saturation Monitor) ---
            # 1. Real-time duration
            if hasattr(self, "_current_sat_duration") and self._current_sat_duration > 0:
                sat_ms_disp = self._current_sat_duration * 1000.0
                thresh_ms = float(self.settings.get("sat_time_thresh_ms", 180.0))
                
                # Color changes as we approach threshold
                c_sat = (0, 255, 255) # Yellow
                if sat_ms_disp > thresh_ms: c_sat = (0, 0, 255) # Red (Counting State)
                
                txt_sat = f"SATURATED: {sat_ms_disp:.1f} ms"
                cv.putText(canvas, txt_sat, (W_win - 350, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, c_sat, 1)

                # Draw simple progress bar
                bar_len = int( min(sat_ms_disp, 300) ) # Max 300px
                cv.rectangle(canvas, (W_win - 350, 55), (W_win - 350 + bar_len, 60), c_sat, -1)
                # Draw threshold marker on bar
                marker_x = int(thresh_ms)
                if marker_x < 300:
                    cv.line(canvas, (W_win - 350 + marker_x, 52), (W_win - 350 + marker_x, 63), (255, 255, 255), 1)

            # 2. Last N1 Event
            if hasattr(self, "_last_n1_event"):
                ts, direction, dur = self._last_n1_event
                sign_str = "+1" if direction > 0 else "-1"
                # Format time
                t_str = time.strftime("%H:%M:%S", time.localtime(ts))
                # Fade out old events? No, just keep last one.
                info_color = (0, 255, 0) if direction > 0 else (0, 100, 255) # Green / Orange
                
                txt_evt = f"Last N1: {sign_str} @ {t_str} (Dur: {dur*1000:.1f}ms)"
                cv.putText(canvas, txt_evt, (W_win - 350, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)

            if profile.size >= 2:
                p = profile.astype(np.float32)
                p_min, p_max = np.min(p), np.max(p)
                if p_max > p_min:
                    p = (p - p_min) / (p_max - p_min)
                    pts = []
                    for i in range(len(p)):
                        x = int(i * W_win / len(p))
                        y = int((h_plot - 10) - p[i] * (h_plot - 40))
                        pts.append((x, y))
                    cv.polylines(canvas, [np.array(pts)], False, (0, 255, 255), 1, cv.LINE_AA)

            # --- 2. FFT (Middle) ---
            cv.rectangle(canvas, (0, h_plot), (W_win, 2 * h_plot), (30, 30, 30), -1)
            cv.line(canvas, (0, h_plot), (W_win, h_plot), (100, 100, 100), 1)
            cv.putText(canvas, "Frequency Domain (FFT)", (10, h_plot + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            if "mag_debug" in locals() and mag_debug is not None and mag_debug.size > 0:
                m = mag_debug.astype(np.float32)
                # Show mainly low frequencies where the fringe is
                limit = min(len(m), 64) 
                m = m[:limit]
                m_max = np.max(m)
                if m_max > 1e-6:
                    m = m / m_max
                
                bin_w = float(W_win) / float(limit)
                for i in range(len(m)):
                    x0 = int(i * bin_w)
                    x1 = int((i + 1) * bin_w) - 1
                    bar_h = int(m[i] * (h_plot - 40))
                    y_base = 2 * h_plot - 5
                    
                    color = (255, 100, 100)
                    if self._k_peak is not None and i == self._k_peak:
                        color = (100, 255, 100)
                        # Label peak
                        cv.putText(canvas, f"k={i}", (x0, y_base - bar_h - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                    
                    cv.rectangle(canvas, (x0, y_base), (x1, y_base - bar_h), color, -1)

            # --- 3. Phase Accumulation (Bottom) ---
            cv.rectangle(canvas, (0, 2 * h_plot), (W_win, 3 * h_plot), (30, 30, 30), -1)
            cv.line(canvas, (0, 2 * h_plot), (W_win, 2 * h_plot), (100, 100, 100), 1)
            
            # --- Update Phase History ---
            current_total_phase = float(self.count_value) + self._u_shift_accum
            if not hasattr(self, "_phase_plot_history"):
                self._phase_plot_history = []
            
            # --- Robust Data Collection Logic for Fitting ---
            # Criteria:
            # 1. External status received.
            # 2. Position MUST change (otherwise lambda = phase_change / 0 = inf).
            # 3. Phase direction consistency check (to handle B/L).
            
            if self._external_status_received:
                now = time.time()
                
                # --- State Initialization ---
                if not hasattr(self, "_last_fit_time"): self._last_fit_time = now
                # If first run, assume prev pos is current
                if not hasattr(self, "_prev_frame_pos"): self._prev_frame_pos = self._stage_position
                
                # --- Movement Detection (Frame-to-Frame) ---
                frame_move = self._stage_position - self._prev_frame_pos
                self._prev_frame_pos = self._stage_position
                
                # --- Data Collection Logic ---
                # Only collect when stage is physically moving
                MOVE_THRESH = 1e-6 
                TIMEOUT_SEC = 2.0
                
                if abs(frame_move) > MOVE_THRESH:
                    current_dir = 1 if frame_move > 0 else -1
                    
                    # 1. Check for Timeout (New Experiment)
                    # If we haven't added data for > TIMEOUT_SEC, assume new move
                    if (now - self._last_fit_time) > TIMEOUT_SEC:
                        if len(self._fit_data) > 0:
                             self._fit_data = [] # Reset on new start
                        self._fit_active_direction = current_dir

                    # 2. Check for Direction Reversal
                    if len(self._fit_data) > 0 and self._fit_active_direction != 0:
                        if current_dir != self._fit_active_direction:
                            self._fit_data = [] # Reset on flip
                            self._fit_active_direction = current_dir
                    
                    # 3. First point init
                    if len(self._fit_data) == 0:
                        self._fit_active_direction = current_dir

                    # 4. Append Data (Avoid duplicate positions & Handle Backlash)
                    last_saved = self._fit_data[-1][0] if len(self._fit_data) > 0 else (self._stage_position - 1.0)
                    
                    # Use the N1 (Saturation Integer) Method for Wavelength Analysis
                    current_fit_N = float(self.count_intensity_method)

                    if abs(self._stage_position - last_saved) > MOVE_THRESH:
                         # --- Backlash / Hysteresis Detection ---
                         # Only check at the start of a movement segment (first ~20 points)
                         is_stuck_in_backlash = False
                         if 0 < len(self._fit_data) < 30:
                             start_pt = self._fit_data[0]
                             d_pos_total = self._stage_position - start_pt[0]
                             d_phase_total = current_fit_N - start_pt[1]
                             
                             # Only judge if we have moved enough to expect a signal (1.0 micron)
                             # 1 um should give ~3 fringes.
                             if abs(d_pos_total) > 0.001: 
                                 # Calculate localized slope: fringes / mm
                                 # Expected ~3160. If we see < 500, it's virtually flat -> Backlash
                                 current_slope = abs(d_phase_total / d_pos_total)
                                 if current_slope < 500:
                                     is_stuck_in_backlash = True
                         
                         if is_stuck_in_backlash:
                             # The points we collected so far were just backlash (stage moved, fringes didn't)
                             # Reset start point to NOW.
                             self._fit_data = [(self._stage_position, current_fit_N)]
                             self._last_fit_time = now
                         else:
                             self._fit_data.append((self._stage_position, current_fit_N))
                             self._last_fit_time = now

                # Limit buffer size (sliding window)
                if len(self._fit_data) > 5000:
                    self._fit_data.pop(0)

            self._phase_plot_history.append(current_total_phase)
            if len(self._phase_plot_history) > 300: # Keep last 300 frames
                 self._phase_plot_history = self._phase_plot_history[-300:]
            
            phase_val = delta_fringe_dbg if "delta_fringe_dbg" in locals() else 0.0
            
            # Left align text
            cv.putText(canvas, f"Phase Trace (Total Fringes) | Delta F: {phase_val:+.4f}", 
                       (10, 2 * h_plot + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1)

            if len(self._phase_plot_history) > 1:
                hist_arr = np.array(self._phase_plot_history, dtype=np.float32)
                h_min = np.min(hist_arr)
                h_max = np.max(hist_arr)
                h_span = h_max - h_min
                if h_span < 0.1: h_span = 0.1 
                
                h_min -= h_span * 0.1
                h_max += h_span * 0.1
                h_span = h_max - h_min
                
                step_x = float(W_win) / 300.0
                if len(hist_arr) < 300:
                    step_x = float(W_win) / 300.0
                else: 
                    step_x = float(W_win) / float(len(hist_arr) - 1)

                pts_phase = []
                for i in range(len(hist_arr)):
                    x = int(i * step_x)
                    val = hist_arr[i]
                    norm_val = (val - h_min) / h_span
                    y = int((3 * h_plot - 10) - norm_val * (h_plot - 40))
                    pts_phase.append((x, y))
                
                if len(pts_phase) > 1:
                    cv.polylines(canvas, [np.array(pts_phase)], False, (255, 100, 255), 2, cv.LINE_AA)
                    
                # Current value marker
                last_pt = pts_phase[-1]
                cv.circle(canvas, last_pt, 4, (255, 255, 255), -1)
                
                text_str = f"{current_total_phase:.3f}"
                # Ensure text is not off screen (left align if near right edge)
                t_x = last_pt[0] - 20
                if t_x > W_win - 80: t_x = W_win - 100
                cv.putText(canvas, text_str, (t_x, last_pt[1] - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.imshow("Fringe Analysis (Profile | FFT | Phase Trace)", canvas)
            
            # --- New Wavelength Analysis Window (Throttled) ---
            now = time.time()
            if now - self._last_draw_timestamp > 0.1: # Max 10 FPS
                self._draw_wavelength_window()
                self._last_draw_timestamp = now

            # Removed cv.waitKey(1) to prevent double event pumping and lag

        self.last_output_frame = output.copy()
        return output

    def _draw_wavelength_window(self):
        # Always create canvas to ensure window appears
        H, W = 400, 600
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        
        # Margins
        m_l, m_r, m_t, m_b = 60, 20, 40, 40
        
        # Early check for insufficient data
        if len(self._fit_data) < 5:
             cv.putText(canvas, "Waiting for stable movement...", (50, H//2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
             cv.putText(canvas, f"Points: {len(self._fit_data)}/5", (50, H//2 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
             cv.imshow("Wavelength Analysis (Phase vs Position)", canvas)
             return
            
        points = np.array(self._fit_data)
        xs = points[:, 0] # mm
        ys = points[:, 1] # fringes
        
        dx = np.max(xs) - np.min(xs)
        if dx < 0.0001: # Reduced threshold to allow earlier feedback
             cv.putText(canvas, "Insufficient displacement...", (50, H//2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
             cv.imshow("Wavelength Analysis (Phase vs Position)", canvas)
             return

        # Fit Line: fringe = m * pos + c
        # slope m = fringes / mm = 2 / lambda
        try:
             # Use Polyfit
             coef = np.polyfit(xs, ys, 1)
             slope = coef[0]
             intercept = coef[1]
             
             # Calculate R^2 or error?
             # lambda = 2 / slope (if slope is positive/negative doesn't matter for magnitude)
             # Careful: if 'spit' is positive, and we move stage 'away', phase increases.
             # Relation depends on setup. We take absolute slope.
             
             meas_lambda_mm = 0.0
             if abs(slope) > 1e-4:
                 meas_lambda_mm = 2.0 / abs(slope)
             
             meas_lambda_nm = meas_lambda_mm * 1e6
             
             # Relative error
             ref_nm = getattr(self, "_reference_wavelength_nm", 632.8)
             error_pct = 0.0
             if ref_nm > 0:
                 error_pct = 100.0 * abs(meas_lambda_nm - ref_nm) / ref_nm
             
             # --- Plotting ---
             x_min, x_max = np.min(xs), np.max(xs)
             y_min, y_max = np.min(ys), np.max(ys)
             
             # Pad
             x_rng = x_max - x_min if x_max > x_min else 1.0
             y_rng = y_max - y_min if y_max > y_min else 1.0
             
             x_min -= x_rng * 0.1
             x_max += x_rng * 0.1
             y_min -= y_rng * 0.1
             y_max += y_rng * 0.1
             
             scale_x = float(W - m_l - m_r) / (x_max - x_min)
             scale_y = float(H - m_t - m_b) / (y_max - y_min)
             
             # Draw Axes
             cv.rectangle(canvas, (m_l, m_t), (W - m_r, H - m_b), (50, 50, 50), 1)
             
             # Plot Points
             # Downsample for drawing if too many
             draw_points = points
             if len(points) > 500:
                  indices = np.linspace(0, len(points)-1, 500).astype(int)
                  draw_points = points[indices]
                  
             pts = []
             for pt in draw_points:
                 px = int(m_l + (pt[0] - x_min) * scale_x)
                 py = int((H - m_b) - (pt[1] - y_min) * scale_y)
                 pts.append((px, py))
                 cv.circle(canvas, (px, py), 2, (0, 255, 255), -1)
             
             # Draw Fit Line
             x0, x1 = x_min, x_max
             y0_fit = slope * x0 + intercept
             y1_fit = slope * x1 + intercept
             
             px0 = int(m_l + (x0 - x_min) * scale_x)
             py0 = int((H - m_b) - (y0_fit - y_min) * scale_y)
             px1 = int(m_l + (x1 - x_min) * scale_x)
             py1 = int((H - m_b) - (y1_fit - y_min) * scale_y)
             
             cv.line(canvas, (px0, py0), (px1, py1), (0, 100, 255), 2, cv.LINE_AA)
             
             # Info Text
             cv.putText(canvas, f"Wavelength Calc (N1 Saturation Fit)", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
             
             # Fit Equation
             fit_eq = f"N = {slope:.1f} * d + {intercept:.1f}"
             cv.putText(canvas, fit_eq, (W - 300, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

             res_color = (0, 255, 0)
             if error_pct > 1.0: res_color = (0, 255, 255)
             if error_pct > 5.0: res_color = (0, 0, 255)
             
             info1 = f"Meas Lambda: {meas_lambda_nm:.2f} nm"
             info2 = f"Ref: {ref_nm:.1f} nm | Error: {error_pct:.2f}%"
             
             cv.putText(canvas, info1, (m_l + 10, m_t + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, res_color, 2)
             cv.putText(canvas, info2, (m_l + 10, m_t + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
             
             # Axis Labels
             cv.putText(canvas, f"{x_min:.3f}", (m_l, H - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
             cv.putText(canvas, f"{x_max:.3f} mm", (W - m_r - 60, H - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
             
             cv.putText(canvas, f"{y_min:.1f}", (5, H - m_b), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
             cv.putText(canvas, f"{y_max:.1f} N", (5, m_t), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

             cv.imshow("Wavelength Analysis (Phase vs Position)", canvas)
             
        except Exception as e:
             pass


try:  # SciPy gives robust peak picking when available.
    from scipy.signal import find_peaks as _scipy_find_peaks  # type: ignore
except Exception:  # pragma: no cover
    _scipy_find_peaks = None

Histogram = np.ndarray
Contour = np.ndarray
BBox = Tuple[int, int, int, int]


def _find_hist_peaks(hist: Histogram) -> np.ndarray:
    """Return peak indices from a histogram."""
    if _scipy_find_peaks is not None:
        peaks, _ = _scipy_find_peaks(hist)
        return peaks
    peaks: List[int] = []
    for idx in range(1, len(hist) - 1):
        if hist[idx] > hist[idx - 1] and hist[idx] > hist[idx + 1]:
            peaks.append(idx)
    return np.array(peaks, dtype=np.int32)


def calculate_threshold(red_channel: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    """Histogram based threshold copied from the reference repo."""
    if mask is not None and mask.shape == red_channel.shape:
        red_channel = cv.bitwise_and(red_channel, red_channel, mask=mask)
    hist = cv.calcHist([red_channel], [0], None, [256], [0, 256]).flatten()
    hist[0] = 0
    peaks = _find_hist_peaks(hist)
    sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
    if len(sorted_peaks) < 2:
        raise ValueError("unable to locate two histogram peaks")
    second_peak_index = min(sorted_peaks[:2])
    min_val_after_peak = int(np.argmin(hist[second_peak_index:]) + second_peak_index)
    return min_val_after_peak


def segment_image(src: np.ndarray, kernel_size: Tuple[int, int]) -> Tuple[BBox, Optional[np.ndarray]]:
    _, src_bin = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    src_bin = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    coords = cv.findNonZero(src_bin)
    if coords is None:
        return (0, 0, 0, 0), None
    contours, _ = cv.findContours(src_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        x, y, w, h = cv.boundingRect(coords)
        return (x, y, w, h), None
    contour = max(contours, key=cv.contourArea)
    mask = np.zeros(src.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 1, thickness=cv.FILLED)
    x, y, w, h = cv.boundingRect(coords)
    return (x, y, w, h), mask


def crop_to_bbox(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w].copy()


def classify_contour(contour: Contour, binary_image: np.ndarray, thickness: int = 20) -> Tuple[int, int, int]:
    mask = np.zeros_like(binary_image)
    cv.drawContours(mask, [contour], -1, 255, thickness)
    filled = np.zeros_like(binary_image)
    cv.drawContours(filled, [contour], -1, 255, thickness=cv.FILLED)
    combined = cv.bitwise_and(mask, filled)
    masked_points = binary_image[combined == 255]
    if len(masked_points) == 0:
        return (255, 255, 255)
    avg_color = np.mean(masked_points)
    return (0, 255, 0) if avg_color < 128 else (255, 0, 0)


@dataclass
class RingObservation:
    contour: Contour
    area: float
    centroid: Tuple[float, float]
    radius: float
    color: Tuple[int, int, int]
    global_centroid: Tuple[float, float]


@dataclass
class RingTrack:
    ring_id: int
    area: float
    centroid: Tuple[float, float]
    radius: float
    color: Tuple[int, int, int]
    last_seen_frame: int
    radius_history: List[float] = field(default_factory=list)
    state: str = "unknown"  # "inside", "outside", "unknown"

    @property
    def smoothed_radius(self) -> float:
        if not self.radius_history:
            return self.radius
        return sum(self.radius_history) / len(self.radius_history)


@dataclass
class ProcessedFrame:
    display_frame: Optional[np.ndarray]
    debug_frame: Optional[np.ndarray]
    bbox_area: float
    contours_info: List[Tuple[float, Tuple[int, int, int]]] = field(default_factory=list)
    center_color: Optional[int] = None
    center_intensity: Optional[float] = None
    center_is_bright: bool = False
    rings: List[RingObservation] = field(default_factory=list)
    bbox_origin: Tuple[int, int] = (0, 0)
    max_possible_radius: float = 0.0


def process_frame(
    frame: np.ndarray,
    previous_bbox_area: float = 0.1,
    kernel_size: Tuple[int, int] = (9, 9),
    center_color_threshold: int = 0,
    min_length: int = 400,
    circularity_threshold: float = 0.7,
    area_threshold: int = 1000,
    region_size: int = 8,
    center_circle_diameter: int = 50,
    center_circle_thickness: int = 2,
    debug: bool = True,
    fill_contours: bool = True,
    fill_alpha: float = 0.5,
    threshold_offset: int = 0,
    binary_mode: str = "dark",
    dark_ratio_min: float = 0.1,
    bright_ratio_min: float = 0.1,
    intensity_margin: int = 15,
    ring_min_radius: float = 0.0,
    ring_min_area: float = 0.0,
) -> ProcessedFrame:
    if frame.ndim == 2:
        # 如果是单通道（灰度）图像，直接使用
        red_channel = frame
    else:
        # 如果是多通道图像，取红色通道（通常对比度较高）
        red_channel = frame[:, :, 2]

    bbox, mask = segment_image(red_channel, (5, 5))
    current_bbox_area = bbox[2] * bbox[3]
    if previous_bbox_area > 0 and current_bbox_area > previous_bbox_area * 1.5:
        bbox, mask = segment_image(red_channel, (7, 7))
        current_bbox_area = bbox[2] * bbox[3]
    if current_bbox_area == 0:
        return ProcessedFrame(None, None, previous_bbox_area, [], None, None, False, [], (0, 0), 0.0)
    bbox_origin = (bbox[0], bbox[1])
    
    red_channel = crop_to_bbox(red_channel, bbox)
    if mask is not None:
        mask = crop_to_bbox(mask, bbox)
    
    cropped_frame = crop_to_bbox(frame, bbox)
    
    try:
        threshold = calculate_threshold(red_channel, mask)
    except ValueError:
        threshold, _ = cv.threshold(red_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    final_threshold = int(np.clip(threshold + threshold_offset, 0, 255))
    thresh_type = cv.THRESH_BINARY_INV if binary_mode == "dark" else cv.THRESH_BINARY
    _, binary_image = cv.threshold(red_channel, final_threshold, 255, thresh_type)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    processed_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    processed_image = cv.morphologyEx(processed_image, cv.MORPH_CLOSE, kernel)
    fill_alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    base_alpha = 1.0 - fill_alpha
    dark_ratio_min = float(np.clip(dark_ratio_min, 0.0, 1.0))
    bright_ratio_min = float(np.clip(bright_ratio_min, 0.0, 1.0))
    intensity_margin = int(max(0, intensity_margin))
    ring_min_radius = float(max(0.0, ring_min_radius))
    ring_min_area = float(max(0.0, ring_min_area))
    
    # 构造彩色显示底图：一份是原图覆盖，一份供调试使用
    if frame.ndim == 2:
        display_frame = cv.cvtColor(cropped_frame, cv.COLOR_GRAY2BGR)
    else:
        display_frame = cropped_frame.copy()

    debug_frame = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR)

    edges = cv.Canny(processed_image, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_radius = min(bbox[2:]) // 2
    max_area = math.pi * (max_radius ** 2)
    contours = sorted(contours, key=cv.contourArea)
    contours_info: List[Tuple[float, Tuple[int, int, int]]] = []
    ring_observations: List[RingObservation] = []
    pre_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter < min_length or area > max_area:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter + 1e-6))
        if circularity >= circularity_threshold:
            if pre_area != 0 and abs(area - pre_area) < area_threshold:
                continue
            pre_area = area
            color = classify_contour(contour, binary_image)
            # 只保留目标明暗的轮廓
            contour_mask = np.zeros_like(binary_image)
            cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
            ring_mask = cv.bitwise_and(contour_mask, processed_image)
            ring_pixels = cv.countNonZero(ring_mask)
            contour_pixels = cv.countNonZero(contour_mask)
            if ring_pixels == 0 or contour_pixels == 0:
                continue
            ring_ratio = ring_pixels / contour_pixels
            ring_mean = cv.mean(red_channel, mask=ring_mask)[0]
            if binary_mode == "dark":
                if ring_ratio < dark_ratio_min:
                    continue
                if ring_mean >= final_threshold - intensity_margin:
                    continue
                color = (0, 255, 0)
            else:
                if ring_ratio < bright_ratio_min:
                    continue
                if ring_mean <= final_threshold + intensity_margin:
                    continue
                color = (255, 0, 0)
            moments = cv.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid_x = float(moments["m10"] / moments["m00"])
            centroid_y = float(moments["m01"] / moments["m00"])
            radius = float(math.sqrt(area / math.pi))
            if radius < ring_min_radius or area < ring_min_area:
                continue
            global_centroid = (
                centroid_x + bbox_origin[0],
                centroid_y + bbox_origin[1],
            )
            contours_info.append((area, color))
            if fill_contours:
                overlay = display_frame.copy()
                overlay[ring_mask == 255] = color
                cv.addWeighted(overlay, fill_alpha, display_frame, base_alpha, 0, display_frame)
                cv.drawContours(display_frame, [contour], -1, color, 2)
                debug_overlay = debug_frame.copy()
                debug_overlay[ring_mask == 255] = color
                cv.addWeighted(debug_overlay, fill_alpha, debug_frame, base_alpha, 0, debug_frame)
                cv.drawContours(debug_frame, [contour], -1, color, 2)
            else:
                cv.drawContours(display_frame, [contour], -1, color, 2)
                cv.drawContours(debug_frame, [contour], -1, color, 2)
            ring_observations.append(
                RingObservation(
                    contour=contour,
                    area=area,
                    centroid=(centroid_x, centroid_y),
                    radius=radius,
                    color=color,
                    global_centroid=global_centroid,
                )
            )
    
    if not ring_observations:
        return ProcessedFrame(
            display_frame,
            debug_frame,
            current_bbox_area,
            contours_info,
            None,
            None,
            False,
            [],
            bbox_origin,
            float(min(cropped_frame.shape[:2])) / 2.0,
        )

    cx = int(sum(obs.centroid[0] for obs in ring_observations) / len(ring_observations))
    cy = int(sum(obs.centroid[1] for obs in ring_observations) / len(ring_observations))

    cy = max(0, min(binary_image.shape[0] - 1, cy))
    cx = max(0, min(binary_image.shape[1] - 1, cx))

    # 中心过滤步骤已按要求禁用：不再基于中心覆盖率丢弃任何环
    # 保留原始 ring_observations，后续只用于采样中心亮暗和可视化

    region = binary_image[
        max(0, cy - region_size) : min(binary_image.shape[0], cy + region_size),
        max(0, cx - region_size) : min(binary_image.shape[1], cx + region_size),
    ]
    avg_color = float(np.mean(region)) if region.size else 128.0
    brightness_threshold = min(255.0, 128.0 + center_color_threshold)
    darkness_threshold = max(0.0, 128.0 - center_color_threshold)
    center_is_bright = avg_color >= brightness_threshold
    if avg_color <= darkness_threshold:
        center_color = 255
        border_color = (0, 255, 0)
    elif avg_color >= brightness_threshold:
        center_color = 0
        border_color = (255, 0, 0)
    else:
        center_color = None
        border_color = (255, 255, 255)
    cv.circle(
        display_frame,
        (cx, cy),
        center_circle_diameter // 2,
        border_color,
        center_circle_thickness,
    )
    cv.circle(
        debug_frame,
        (cx, cy),
        center_circle_diameter // 2,
        border_color,
        center_circle_thickness,
    )
    # 画出中心采样点，方便调试
    cv.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)
    cv.circle(debug_frame, (cx, cy), 2, (0, 255, 255), -1)
    
    return ProcessedFrame(
        display_frame,
        debug_frame,
        current_bbox_area,
        contours_info,
        center_color,
        avg_color,
        center_is_bright,
        ring_observations,
        bbox_origin,
        float(min(cropped_frame.shape[:2])) / 2.0,
    )


@dataclass
class ContourDetection:
    settings: Optional[dict] = None

    def __post_init__(self) -> None:
        defaults = {
            "kernel_size": (5, 5),  # 稍微增大核大小以平滑噪点
            "center_color_threshold": 0,
            "min_length": 50,       # 再次降低长度阈值以提高灵敏度
            "circularity_threshold": 0.2, # 再次降低圆形度阈值
            "area_threshold": 1000,
            "region_size": 4,       # 稍微增大采样区域
            "center_circle_diameter": 50,
            "center_circle_thickness": 2,
            "fps": 10,
            "mode": "zoom_in",
            "debug": False,
            "threshold_offset": 0,
            "binary_mode": "dark",  # dark 仅识别暗条纹，bright 仅识别亮条纹
            "fill_contours": True,
            "fill_alpha": 0.5,
            "dark_ratio_min": 0.1,
            "bright_ratio_min": 0.1,
            "intensity_margin": 15,
            "track_max_distance": 40.0,
            "track_max_area_change": 0.35,
            "track_max_inactive": 5,
            "ring_min_radius": 12.0,
            "ring_min_area": 500.0,
            "ring_growth_ratio": 1.3,
            "ring_growth_delta": 40.0,
            "ring_shrink_ratio": 0.75,
            "ring_shrink_delta": 40.0,
            "ring_confirm_frames": 1,
            "ring_event_display_frames": 30,
            "process_stride": 2,
            "hysteresis_upper": 0.70,
            "hysteresis_lower": 0.60,
        }
        self.settings = {**defaults, **(self.settings or {})}
        self.reset_counters()
        self.reset_state()

    def reset_counters(self) -> None:
        self.green_count = 0
        self.blue_count = 0
        self.white_count = 0
        self.count_value = 0
        self.total_frames = 0

    def reset_state(self) -> None:
        self.previous_bbox_area = 0.0
        self.previous_contours_info: List[Tuple[float, Tuple[int, int, int]]] = []
        self.previous_center_color: Optional[int] = None
        self.status: Optional[str] = None
        self.ring_tracks: Dict[int, RingTrack] = {}
        self.previous_tracks_snapshot: Dict[int, RingTrack] = {}
        self.next_ring_id = 1
        self.previous_ring_count = 0
        self.previous_outer_radius = 0.0
        self.previous_inner_radius = 0.0
        self.new_ring_persistence = 0
        self.lost_ring_persistence = 0
        self.recent_ring_event: Optional[str] = None
        self.recent_ring_event_timer = 0
        self.last_output_frame: Optional[np.ndarray] = None
        self.latest_swallow_spit_debug: Dict[str, float | int | str | None | bool] = {}

    def toggle_debug(self) -> None:
        current = self.settings.get("debug", False)
        self.settings["debug"] = not current
        mode = "调试" if self.settings["debug"] else "原始"
        print(f"[ContourDetection] 已切换至{mode}显示模式")

    def _match_rings(self, rings: List[RingObservation]) -> Dict[int, RingTrack]:
        max_distance = float(self.settings.get("track_max_distance", 40.0))
        max_area_change = float(self.settings.get("track_max_area_change", 0.35))
        max_inactive = int(self.settings.get("track_max_inactive", 5))
        updated_tracks: Dict[int, RingTrack] = {}
        used_track_ids: Set[int] = set()

        # 清理长时间未出现的轨迹
        active_tracks: Dict[int, RingTrack] = {}
        for track_id, track in self.ring_tracks.items():
            if self.total_frames - track.last_seen_frame <= max_inactive:
                active_tracks[track_id] = track
        self.ring_tracks = active_tracks

        for ring in sorted(rings, key=lambda r: r.radius):
            best_track_id: Optional[int] = None
            best_score = float("inf")
            for track_id, track in self.ring_tracks.items():
                if track_id in used_track_ids:
                    continue
                center_dist = math.hypot(
                    ring.global_centroid[0] - track.centroid[0],
                    ring.global_centroid[1] - track.centroid[1],
                )
                if center_dist > max_distance:
                    continue
                area_ratio = abs(ring.area - track.area) / max(track.area, 1.0)
                if area_ratio > max_area_change:
                    continue
                score = center_dist + area_ratio * max_distance
                if score < best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_ring_id
                self.next_ring_id += 1
                history = []
                state = "unknown"
            else:
                track_id = best_track_id
                used_track_ids.add(best_track_id)
                prev_track = self.ring_tracks[track_id]
                history = prev_track.radius_history[-9:]
                state = prev_track.state

            history.append(ring.radius)

            updated_tracks[track_id] = RingTrack(
                ring_id=track_id,
                area=ring.area,
                centroid=ring.global_centroid,
                radius=ring.radius,
                color=ring.color,
                last_seen_frame=self.total_frames,
                radius_history=history,
                state=state,
            )

        self.ring_tracks = updated_tracks
        return updated_tracks

    def update(self, frame: np.ndarray) -> np.ndarray:
        self.total_frames += 1
        stride = max(1, int(self.settings.get("process_stride", 1)))
        should_process = (
            stride == 1
            or self.last_output_frame is None
            or ((self.total_frames - 1) % stride == 0)
        )
        if not should_process and self.last_output_frame is not None:
            return self.last_output_frame.copy()

        result = process_frame(
            frame,
            self.previous_bbox_area,
            self.settings["kernel_size"],
            self.settings["center_color_threshold"],
            self.settings["min_length"],
            self.settings["circularity_threshold"],
            self.settings["area_threshold"],
            self.settings["region_size"],
            self.settings["center_circle_diameter"],
            self.settings["center_circle_thickness"],
            self.settings.get("debug", False),
            self.settings.get("fill_contours", True),
            self.settings.get("fill_alpha", 0.5),
            self.settings.get("threshold_offset", 0),
            self.settings.get("binary_mode", "dark"),
            self.settings.get("dark_ratio_min", 0.1),
            self.settings.get("bright_ratio_min", 0.1),
            self.settings.get("intensity_margin", 15),
            self.settings.get("ring_min_radius", 12.0),
            self.settings.get("ring_min_area", 500.0),
        )

        # 准备输出画布：如果是灰度图，转为 BGR 以便显示彩色信息（边框、文字等）
        if frame.ndim == 2:
            output_canvas = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            output_canvas = frame.copy()

        if result.display_frame is None:
            self.white_count += 1
            return output_canvas

        display_frame = result.display_frame
        debug_frame = result.debug_frame

        prev_tracks_snapshot = self.previous_tracks_snapshot
        tracked_rings = self._match_rings(result.rings)
        current_tracks_snapshot = {tid: track for tid, track in tracked_rings.items()}
        bbox_origin = result.bbox_origin
        sorted_tracks = sorted(tracked_rings.values(), key=lambda t: t.radius)
        current_ring_count = len(sorted_tracks)
        ring_summary_lines = [f"count: {current_ring_count}"]
        ring_summary_lines += [f"#{track.ring_id}: r={track.radius:.1f}px" for track in sorted_tracks]
        current_outer_radius = sorted_tracks[-1].radius if sorted_tracks else 0.0
        current_inner_radius = sorted_tracks[0].radius if sorted_tracks else 0.0

        growth_ratio = float(self.settings.get("ring_growth_ratio", 1.3))
        growth_delta = float(self.settings.get("ring_growth_delta", 40.0))
        shrink_ratio = float(self.settings.get("ring_shrink_ratio", 0.75))
        shrink_delta = float(self.settings.get("ring_shrink_delta", 40.0))
        confirm_frames = max(1, int(self.settings.get("ring_confirm_frames", 3)))

        ring_event: Optional[str] = None
        
        # New Hysteresis Logic
        max_possible_radius = result.max_possible_radius
        hysteresis_upper = float(self.settings.get("hysteresis_upper", 0.90))
        hysteresis_lower = float(self.settings.get("hysteresis_lower", 0.80))
        
        if max_possible_radius > 0:
            r_out = max_possible_radius * hysteresis_upper
            r_in = max_possible_radius * hysteresis_lower
            
            for track in tracked_rings.values():
                r = track.smoothed_radius
                
                if track.state == "unknown":
                    if r > r_out:
                        track.state = "outside"
                    elif r < r_in:
                        track.state = "inside"
                
                elif track.state == "inside":
                    if r > r_out:
                        track.state = "outside"
                        self.count_value += 1
                        ring_event = "spit"
                        self.status = "zoom_out"
                
                elif track.state == "outside":
                    if r < r_in:
                        track.state = "inside"
                        self.count_value -= 1
                        ring_event = "swallow"
                        self.status = "zoom_in"

        if ring_event is not None:
            self.recent_ring_event = ring_event
            self.recent_ring_event_timer = int(self.settings.get("ring_event_display_frames", 30))
        elif self.recent_ring_event_timer > 0:
            self.recent_ring_event_timer -= 1
            if self.recent_ring_event_timer == 0:
                self.recent_ring_event = None

        # Display info
        status_text = self.status or "idle"
        ring_summary_lines.append(f"status: {status_text}")
        if max_possible_radius > 0:
             ring_summary_lines.append(f"Gate: {r_in:.0f}-{r_out:.0f} px")
             # Show radius of the largest track for debugging
             if sorted_tracks:
                 largest_r = sorted_tracks[-1].smoothed_radius
                 largest_state = sorted_tracks[-1].state
                 ring_summary_lines.append(f"MaxR: {largest_r:.0f} ({largest_state})")
             else:
                 ring_summary_lines.append("No rings tracked")
        
        if self.recent_ring_event is not None:
            ring_summary_lines.append(f"event: {self.recent_ring_event}")

        self.previous_tracks_snapshot = current_tracks_snapshot
        for track in sorted_tracks:
            local_x = int(round(track.centroid[0] - bbox_origin[0]))
            local_y = int(round(track.centroid[1] - bbox_origin[1]))
            if not (0 <= local_x < display_frame.shape[1] and 0 <= local_y < display_frame.shape[0]):
                continue
            label = f"#{track.ring_id} r={track.radius:.0f}"
            cv.circle(display_frame, (local_x, local_y), 4, track.color, -1)
            cv.putText(
                display_frame,
                label,
                (local_x + 6, local_y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                track.color,
                1,
                cv.LINE_AA,
            )
            if debug_frame is not None:
                cv.circle(debug_frame, (local_x, local_y), 4, track.color, -1)
                cv.putText(
                    debug_frame,
                    label,
                    (local_x + 6, local_y - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    track.color,
                    1,
                    cv.LINE_AA,
                )

        self.previous_bbox_area = result.bbox_area
        current_contours_info = result.contours_info
        current_center_color = result.center_color
        center_is_bright = result.center_is_bright

        if ring_event == "spit":
            border_color = (0, 255, 0)
        elif ring_event == "swallow":
            border_color = (255, 0, 0)
        else:
            border_color = (255, 255, 255)

        # 仅用于统计调试帧的颜色计数
        if border_color == (0, 255, 0):
            self.green_count += 1
        elif border_color == (255, 0, 0):
            self.blue_count += 1
        else:
            self.white_count += 1
        border_thickness = 20

        def _make_border(img: np.ndarray) -> np.ndarray:
            return cv.copyMakeBorder(
                img,
                border_thickness,
                border_thickness,
                border_thickness,
                border_thickness,
                cv.BORDER_CONSTANT,
                value=border_color,
            )

        preview_frame = debug_frame if self.settings.get("debug", False) and debug_frame is not None else display_frame
        bordered_preview = _make_border(preview_frame)

        h, w = bordered_preview.shape[:2]
        # 确保不越界
        h = min(h, output_canvas.shape[0])
        w = min(w, output_canvas.shape[1])
        output_canvas[:h, :w] = bordered_preview[:h, :w]

        if ring_summary_lines:
            panel_width = 220
            panel_line_height = 22
            panel_height = 30 + panel_line_height * len(ring_summary_lines)
            panel_x = max(0, output_canvas.shape[1] - panel_width - 20)
            panel_y = 20
            overlay = output_canvas.copy()
            cv.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                (0, 0, 0),
                thickness=cv.FILLED,
            )
            cv.addWeighted(overlay, 0.4, output_canvas, 0.6, 0, output_canvas)
            cv.putText(
                output_canvas,
                "Rings",
                (panel_x + 10, panel_y + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )
            for idx, line in enumerate(ring_summary_lines):
                y = panel_y + 20 + (idx + 1) * panel_line_height
                cv.putText(
                    output_canvas,
                    line,
                    (panel_x + 10, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            self.previous_ring_count = current_ring_count
            self.previous_outer_radius = current_outer_radius
            self.previous_inner_radius = current_inner_radius

        # Adaptive font size and position for the main count number
        h_canvas, w_canvas = output_canvas.shape[:2]
        font_scale = max(1.5, min(w_canvas, h_canvas) / 150.0)
        thickness = max(3, int(font_scale * 3))
        count_text = str(self.count_value)
        
        (text_w, text_h), _ = cv.getTextSize(count_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Position: Top-Right with margin
        text_x = w_canvas - text_w - 30
        text_y = text_h + 30
        
        # Draw black outline for better visibility
        cv.putText(
            output_canvas,
            count_text,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 6,
            cv.LINE_AA
        )
        # Draw white text
        cv.putText(
            output_canvas,
            count_text,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255), # Yellow for high visibility
            thickness,
            cv.LINE_AA
        )

        self.previous_contours_info = current_contours_info
        # 仅在当前颜色有效时更新 previous_center_color，从而忽略中间的过渡态(None)
        if current_center_color is not None:
            self.previous_center_color = current_center_color
        self.last_output_frame = output_canvas.copy()
        return output_canvas

    def process_video(self, input_path: str, output_path: str) -> None:
        cap = cv.VideoCapture(input_path)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("无法打开视频文件")
        original_height, original_width = frame.shape[:2]
        out = cv.VideoWriter(
            output_path,
            cv.VideoWriter_fourcc(*"mp4v"),
            self.settings["fps"],
            (original_width, original_height),
        )
        self.reset_counters()
        self.reset_state()
        while ret:
            frame = self.update(frame)
            out.write(frame)
            ret, frame = cap.read()
        cap.release()
        out.release()
