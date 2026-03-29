"""Utility script to run fringe detection on saved images or videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from fringe_counter import ContourDetection


def _load_image(path: Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法打开图像: {path}")
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return image


def run_image(path: Path, output: Path) -> None:
    detector = ContourDetection()
    frame = _load_image(path)
    processed = detector.update(frame)
    cv.imwrite(str(output), processed)
    print(f"已输出处理结果: {output}")
    print(f"累计计数值: {detector.count_value}")


def run_video(path: Path, output: Path) -> None:
    detector = ContourDetection()
    detector.process_video(str(path), str(output))
    print(f"视频处理完成，结果为: {output}")
    print(f"总圈数: {detector.count_value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Michelson fringe detector")
    parser.add_argument("--image", type=Path, help="静态图像输入路径")
    parser.add_argument("--video", type=Path, help="视频输入路径")
    parser.add_argument("--output", type=Path, default=Path("processed.png"), help="输出路径")
    args = parser.parse_args()

    if args.image and args.video:
        parser.error("图像和视频模式只能二选一")
    if not args.image and not args.video:
        parser.error("需要提供 --image 或 --video 之一")

    if args.image:
        run_image(args.image, args.output)
    else:
        run_video(args.video, args.output)


if __name__ == "__main__":
    main()
