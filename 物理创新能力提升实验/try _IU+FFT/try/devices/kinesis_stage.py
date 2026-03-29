"""Thorlabs Kinesis stage control wrapper using pythonnet.

This module loads Kinesis .NET assemblies from the local devices/Kinesis folder
and exposes a thin Pythonic wrapper to enumerate devices and control a stage.

Tested pattern: KCube/Benchtop Stepper/DC Servo families via their *CLI DLLs.
Actual method names vary slightly across models; the wrapper adapts best-effort.

Prerequisites
- Windows + official Thorlabs Kinesis runtime/drivers
- Python deps: pythonnet (already in requirements.txt)
- Kinesis DLLs available under devices/Kinesis (bundled in this repo)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import os
import shutil
import sys
from contextlib import contextmanager
from typing import Iterable


# Lazy-load pythonnet only on Windows
def _ensure_pythonnet() -> None:
    try:
        import clr  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"pythonnet Check Failed: {exc}. sys.path: {sys.path}"
        ) from exc


def _add_reference(dll_path: Path) -> None:
    import clr  # type: ignore

    if not dll_path.exists():
        return
    try:
        clr.AddReference(str(dll_path))
    except Exception as _:
        # 某些 DLL 可能需要先加载其他依赖，失败时忽略，后续会再次尝试
        pass


def _load_kinesis_assemblies(root: Path) -> None:
    """Load essential Kinesis .NET assemblies by absolute path.

    Load order: common tools -> device manager -> generic -> specific families.
    """
    _ensure_pythonnet()

    # Common toolkits and base libs
    base = [
        "Thorlabs.MotionControl.Tools.Common.dll",
        "Thorlabs.MotionControl.Tools.Logging.dll",
        "Thorlabs.MotionControl.DeviceManager.dll",
        "Thorlabs.MotionControl.DeviceManagerCLI.dll",
        "Thorlabs.MotionControl.GenericMotorCLI.dll",
    ]

    # Family-specific CLI layers we may need. List common ones.
    families = [
        # KCube
        "Thorlabs.MotionControl.KCube.StepperMotorCLI.dll",
        "Thorlabs.MotionControl.KCube.DCServoCLI.dll",
        "Thorlabs.MotionControl.KCube.PiezoCLI.dll",
        # Benchtop
        "Thorlabs.MotionControl.Benchtop.StepperMotorCLI.dll",
        "Thorlabs.MotionControl.Benchtop.DCServoCLI.dll",
        "Thorlabs.MotionControl.Benchtop.PiezoCLI.dll",
        # Integrated
        "Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll",
        "Thorlabs.MotionControl.IntegratedPrecisionPiezoCLI.dll",
    ]

    for name in base + families:
        _add_reference(root / name)


def _configure_utf8_io() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _copy_files(src_files: Iterable[Path], dst_dir: Path) -> None:
    for src in src_files:
        if not src.exists():
            continue
        try:
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        except Exception:
            pass


def _ensure_kinesis_settings(root: Path) -> Path:
    """Ensure Kinesis settings XMLs are discoverable by the .NET SDK.

    Returns the ProgramData path used for settings.
    """
    program_data = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    settings_dir = program_data / "Thorlabs" / "Kinesis"
    try:
        settings_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Copy key XMLs (device database + defaults)
    xml_files = list(root.glob("*.xml"))
    _copy_files(xml_files, settings_dir)
    return settings_dir


@contextmanager
def _temp_cwd(path: Path):
    """Temporarily switch working directory to help Kinesis find settings files."""
    old = Path.cwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(str(old))


@dataclass
class StageInfo:
    serial: str
    model: Optional[str] = None


class KinesisStage:
    """Unified wrapper for a single Kinesis motor axis."""

    def __init__(self, kinesis_dir: Optional[Path] = None) -> None:
        if sys.platform != "win32":  # pragma: no cover
            raise RuntimeError("Kinesis 仅支持 Windows 平台运行")
        self.kinesis_dir = kinesis_dir or Path(r"D:\物理\迈克尔逊\devices\Kinesis")
        if not self.kinesis_dir.exists(): 
            # Fallback to local
            self.kinesis_dir = Path(__file__).resolve().parent / "Kinesis"
            
        if not self.kinesis_dir.exists():  # pragma: no cover
             # Last resort, assume user didn't copy folder but has it in referenced path
             self.kinesis_dir = Path(r"D:\物理\迈克尔逊\devices\Kinesis")

        if self.kinesis_dir.exists():
            _load_kinesis_assemblies(self.kinesis_dir)
        else:
            print(f"Warning: Kinesis DLL folder not found at {self.kinesis_dir}")

        # Late imports after assemblies are loaded
        try:
            from Thorlabs.MotionControl.DeviceManagerCLI import (  # type: ignore
                DeviceManagerCLI,
            )
            self._DeviceManagerCLI = DeviceManagerCLI
        except ImportError:
            self._DeviceManagerCLI = None

        self._device = None  # underlying CLI device instance
        self._connected_serial: Optional[str] = None

    # ---------- Discovery ----------
    def list_devices(self) -> List[StageInfo]:
        """Return all motor device serials detected by Kinesis."""
        if not self._DeviceManagerCLI: return []
        self._DeviceManagerCLI.BuildDeviceList()
        serials = list(self._DeviceManagerCLI.GetDeviceList())
        return [StageInfo(serial=s) for s in serials]

    # ---------- Connect ----------
    def connect(self, serial: str, device_type: Optional[str] = None, settings_timeout_ms: int = 5000) -> None:      
        """Connect to a device by serial number."""
        if not self._DeviceManagerCLI: return
        
        self._DeviceManagerCLI.BuildDeviceList()

        os.environ.setdefault("KINESIS_PATH", str(self.kinesis_dir))
        settings_dir = _ensure_kinesis_settings(self.kinesis_dir)
        os.environ.setdefault("KINESIS_DEVICEDATABASE", str(settings_dir))

        candidates: List[Tuple[str, str, str]] = [
            # KCube
            ("Thorlabs.MotionControl.KCube.StepperMotorCLI", "KCubeStepper", "CreateKCubeStepper"),
            ("Thorlabs.MotionControl.KCube.DCServoCLI", "KCubeDCServo", "CreateKCubeDCServo"),
            ("Thorlabs.MotionControl.KCube.PiezoCLI", "KCubePiezo", "CreateKCubePiezo"),
            ("Thorlabs.MotionControl.KCube.InertialMotorCLI", "KCubeInertialMotor", "CreateKCubeInertialMotor"),     
            # Benchtop
            ("Thorlabs.MotionControl.Benchtop.StepperMotorCLI", "BenchtopStepperMotor", "CreateBenchtopStepperMotor"),
            ("Thorlabs.MotionControl.Benchtop.DCServoCLI", "BenchtopDCServo", "CreateBenchtopDCServo"),
            ("Thorlabs.MotionControl.Benchtop.PiezoCLI", "BenchtopPiezo", "CreateBenchtopPiezo"),
            # Integrated
            ("Thorlabs.MotionControl.IntegratedStepperMotorsCLI", "IntegratedStepperMotor", "CreateIntegratedStepperMotor"),
            ("Thorlabs.MotionControl.IntegratedPrecisionPiezoCLI", "IntegratedPrecisionPiezo", "CreateIntegratedPrecisionPiezo"),
            # Generic fallback
            ("Thorlabs.MotionControl.GenericMotorCLI", "GenericMotor", "CreateGenericMotor"),
        ]

        if device_type:
            prefix = device_type.strip().lower()
            candidates = [c for c in candidates if prefix in c[0].lower()]

        last_err: Optional[Exception] = None
        
        # Try finding the device factory
        for module_name, class_name, factory in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                creator = getattr(cls, factory)

                dev = None
                try:
                    dev = creator(serial)
                except Exception:
                    continue
                
                if dev is None: continue

                dev.Connect(serial)
                if hasattr(dev, "WaitForSettingsInitialized"):
                    dev.WaitForSettingsInitialized(settings_timeout_ms)

                if hasattr(dev, "LoadMotorConfiguration"):
                    dev.LoadMotorConfiguration(serial)

                if hasattr(dev, "StartPolling"):
                    dev.StartPolling(200)
                if hasattr(dev, "EnableDevice"):
                    dev.EnableDevice()
                self._device = dev
                self._connected_serial = serial
                return
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"无法连接到设备 {serial}: {last_err}")

    # ---------- Motion ----------
    def home(self, timeout_ms: int = 60_000) -> None:
        dev = self._require_device()
        if hasattr(dev, "Home"):
            dev.Home(timeout_ms)

    def stop(self) -> None:
        dev = self._require_device()
        if hasattr(dev, "Stop"):
            dev.Stop(0) # 0 = StopImmediate

    def set_velocity(self, vel: float, acc: Optional[float] = None) -> None:
        dev = self._require_device()
        
        # Basic implementation for KCube Stepper
        try:
             d_vel = self._to_decimal(vel)
             d_acc = self._to_decimal(acc if acc else vel*10)
             
             if hasattr(dev, "SetVelocityParams"):
                dev.SetVelocityParams(d_vel, d_acc)
        except Exception:
            pass

    def _to_decimal(self, value: float):
        import System
        return System.Decimal.Parse("{:.9f}".format(value))

    def move_to(self, position: float, timeout_ms: int = 60_000, wait: bool = True) -> None:
        dev = self._require_device()
        d_pos = self._to_decimal(position)
        if hasattr(dev, "MoveTo"):
             if wait:
                 dev.MoveTo(d_pos, timeout_ms)
             else:
                 dev.MoveTo(d_pos, 0)

    def move_by(self, delta: float, timeout_ms: int = 60_000, wait: bool = True) -> None:
        dev = self._require_device()
        d_delta = self._to_decimal(delta)
    def move_by(self, delta: float, timeout_ms: int = 60_000, wait: bool = True) -> None:
        dev = self._require_device()
        d_delta = self._to_decimal(delta)
        
        # Try different MoveRelative signatures
        # 1. MoveRelative(Decimal distance, int timeout)
        # 2. MoveRelative(Decimal distance)
        # 3. MoveRelative(MotorDirection dir, Decimal distance)
        
        if hasattr(dev, "MoveRelative"):
             # Detect direction for signature 3
             from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection
             direction = MotorDirection.Forward if delta >= 0 else MotorDirection.Backward
             d_abs_delta = self._to_decimal(abs(delta))

             if wait:
                 try:
                    # Preferred: (Decimal, int)
                    dev.MoveRelative(d_delta, timeout_ms)
                 except Exception:
                    try:
                        # Fallback: (Decimal) -> then poll
                        dev.MoveRelative(d_delta)
                        self._wait_for_stop(dev, timeout_ms)
                    except Exception:
                        try:
                            # Fallback: (Direction, Decimal) -> then poll
                            dev.MoveRelative(direction, d_abs_delta)
                            self._wait_for_stop(dev, timeout_ms)
                        except Exception:
                            # Final Attempt: SetJogStepSize + MoveJog (Uses Jog Velocity Profile)
                            if hasattr(dev, "SetJogStepSize") and hasattr(dev, "MoveJog"):
                                dev.SetJogStepSize(d_abs_delta)
                                dev.MoveJog(direction, timeout_ms)
                            else:
                                raise # Re-raise the last error if we can't Jog
             else:
                 # Non-blocking wanted
                 try:
                     # Preferred: (Decimal, 0) for immediate return?
                     dev.MoveRelative(d_delta, 0)
                 except Exception:
                     try:
                         # Fallback: (Decimal)
                         dev.MoveRelative(d_delta)
                     except Exception:
                         try:
                             # Fallback: (Direction, Decimal)
                             dev.MoveRelative(direction, d_abs_delta)
                         except Exception:
                            if hasattr(dev, "SetJogStepSize") and hasattr(dev, "MoveJog"):
                                dev.SetJogStepSize(d_abs_delta)
                                dev.MoveJog(direction, 0)
                            else:
                                raise
        else:
             # Fallback to absolute move if no relative move exists
             pos = self.get_position()
             pos = self.get_position()
             self.move_to(pos + delta, timeout_ms, wait)

    @property
    def is_moving(self) -> bool:
        dev = self._require_device()
        if hasattr(dev, "Status") and hasattr(dev.Status, "IsAnyMoving"):
             return dev.Status.IsAnyMoving
        return False

    def move_continuous(self, direction: int) -> None:
        """Start continuous move. direction: 1 (Forward) or -1 (Backward)"""
        if direction == 0:
            self.stop()
            return
        dev = self._require_device()
        try:
            from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection
            d = MotorDirection.Forward if direction > 0 else MotorDirection.Backward
            if hasattr(dev, "MoveContinuous"):
                dev.MoveContinuous(d)
        except Exception:
            pass

    def _wait_for_stop(self, dev, timeout_ms):
        import time
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            if hasattr(dev, "Status") and hasattr(dev.Status, "IsAnyMoving"):
                if not dev.Status.IsAnyMoving:
                    return
            time.sleep(0.05)
            
    def get_position(self) -> float:
        dev = self._require_device()
        if hasattr(dev, "Position"):
            val = dev.Position
            return float(str(val))
        if hasattr(dev, "GetPosition"):
            val = dev.GetPosition()
            return float(str(val))
        return 0.0

    def close(self) -> None:
        if self._device is None:
            return
        try:
            if hasattr(self._device, "StopPolling"):
                self._device.StopPolling()
            if hasattr(self._device, "Disconnect"):
                self._device.Disconnect(True)
        except Exception:
            pass
        finally:
            self._device = None
            self._connected_serial = None

    def _require_device(self):
        if self._device is None:
            raise RuntimeError("尚未连接设备")
        return self._device
