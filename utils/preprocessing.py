"""
图像预处理工具模块
包含数据增强、人脸检测预处理、图像标准化等功能
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter


class ImagePreprocessor:
    """图像预处理器类"""

    def __init__(self, target_size=(48, 48)):
        self.target_size = target_size

    def resize_with_ratio(self, image, size=None):
        """
        保持长宽比调整图像大小

        Args:
            image: 输入图像 (numpy array or PIL Image)
            size: 目标大小 (width, height)

        Returns:
            调整后的图像
        """
        if size is None:
            size = self.target_size

        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            target_w, target_h = size

            # 计算缩放比例
            ratio = min(target_w / w, target_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            resized = cv2.resize(image, (new_w, new_h))

            # 填充到目标大小
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            result = cv2.copyMakeBorder(resized, pad_top, pad_bottom,
                                        pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return result
        else:
            return image.resize(size, Image.Resampling.LANCZOS)

    def normalize_image(self, image, method='minmax'):
        """
        图像归一化

        Args:
            image: 输入图像
            method: 归一化方法 ('minmax', 'meanstd', 'tanh')

        Returns:
            归一化后的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = image.astype(np.float32)

        if method == 'minmax':
            # 归一化到 [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif method == 'meanstd':
            # 均值标准差归一化
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)
        elif method == 'tanh':
            # tanh归一化到 [-1, 1]
            image = (image - 127.5) / 127.5

        return image

    def histogram_equalization(self, image):
        """
        直方图均衡化（增强对比度）

        Args:
            image: 输入图像 (灰度图)

        Returns:
            均衡化后的图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.equalizeHist(image)

    def adjust_brightness_contrast(self, image, alpha=1.0, beta=0):
        """
        调整亮度和对比度

        Args:
            image: 输入图像
            alpha: 对比度 (1.0 = 原图)
            beta: 亮度 (0 = 原图)

        Returns:
            调整后的图像
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


class DataAugmentation:
    """数据增强类"""

    def __init__(self, target_size=(48, 48)):
        self.target_size = target_size

    def random_rotation(self, image, max_angle=15):
        """
        随机旋转图像

        Args:
            image: 输入图像
            max_angle: 最大旋转角度

        Returns:
            旋转后的图像
        """
        angle = np.random.uniform(-max_angle, max_angle)

        if isinstance(image, Image.Image):
            return image.rotate(angle, expand=False)
        else:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

    def random_flip(self, image, p=0.5):
        """
        随机水平翻转

        Args:
            image: 输入图像
            p: 翻转概率

        Returns:
            翻转后的图像
        """
        if np.random.random() < p:
            if isinstance(image, Image.Image):
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return cv2.flip(image, 1)
        return image

    def random_brightness(self, image, delta=0.2):
        """
        随机调整亮度

        Args:
            image: 输入图像
            delta: 亮度变化范围 [-delta, delta]

        Returns:
            调整后的图像
        """
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Brightness(image)
            factor = 1 + np.random.uniform(-delta, delta)
            return enhancer.enhance(factor)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = hsv[:, :, 2] * (1 + np.random.uniform(-delta, delta))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def random_contrast(self, image, delta=0.2):
        """
        随机调整对比度

        Args:
            image: 输入图像
            delta: 对比度变化范围 [1-delta, 1+delta]

        Returns:
            调整后的图像
        """
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Contrast(image)
            factor = 1 + np.random.uniform(-delta, delta)
            return enhancer.enhance(factor)
        else:
            alpha = 1 + np.random.uniform(-delta, delta)
            return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    def random_crop(self, image, crop_size=None, p=0.5):
        """
        随机裁剪

        Args:
            image: 输入图像
            crop_size: 裁剪大小 (width, height)
            p: 裁剪概率

        Returns:
            裁剪后的图像
        """
        if np.random.random() > p:
            return image

        if crop_size is None:
            crop_size = self.target_size

        if isinstance(image, Image.Image):
            w, h = image.size
            crop_w, crop_h = crop_size

            if crop_w < w and crop_h < h:
                left = np.random.randint(0, w - crop_w)
                top = np.random.randint(0, h - crop_h)
                return image.crop((left, top, left + crop_w, top + crop_h))
            else:
                return image.resize(crop_size, Image.Resampling.LANCZOS)
        else:
            h, w = image.shape[:2]
            crop_w, crop_h = crop_size

            if crop_w < w and crop_h < h:
                x = np.random.randint(0, w - crop_w)
                y = np.random.randint(0, h - crop_h)
                return image[y:y + crop_h, x:x + crop_w]
            else:
                return cv2.resize(image, crop_size)

    def add_noise(self, image, noise_type='gaussian', intensity=0.05):
        """
        添加噪声

        Args:
            image: 输入图像
            noise_type: 噪声类型 ('gaussian', 'salt_pepper')
            intensity: 噪声强度

        Returns:
            添加噪声后的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy = image + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)

        elif noise_type == 'salt_pepper':
            noisy = image.copy()
            salt_pepper = np.random.random(image.shape)
            noisy[salt_pepper < intensity / 2] = 0
            noisy[salt_pepper > 1 - intensity / 2] = 255
            return noisy

        return image

    def apply_CLAHE(self, image, clip_limit=2.0, grid_size=(8, 8)):
        """
        应用CLAHE（限制对比度自适应直方图均衡化）
        增强局部对比度，特别适合光照不均的人脸图像

        Args:
            image: 输入图像
            clip_limit: 对比度限制
            grid_size: 网格大小

        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(gray)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else enhanced


class FaceAlignment:
    """人脸对齐类"""

    def __init__(self, target_size=(48, 48)):
        self.target_size = target_size

    def align_face(self, image, landmarks=None):
        """
        人脸对齐（基于眼睛坐标）

        Args:
            image: 输入图像
            landmarks: 面部关键点（需要检测器提供）

        Returns:
            对齐后的人脸
        """
        if landmarks is None:
            # 如果没有关键点，返回原图
            return cv2.resize(image, self.target_size)

        # 获取左右眼中心点（需要实际的关键点索引）
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)

        # 计算旋转角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # 旋转图像
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        return cv2.resize(rotated, self.target_size)

    def center_face(self, face_roi, margin_ratio=0.2):
        """
        居中裁剪人脸，添加边距

        Args:
            face_roi: 人脸区域
            margin_ratio: 边距比例

        Returns:
            居中后的人脸
        """
        h, w = face_roi.shape[:2]
        margin_h = int(h * margin_ratio)
        margin_w = int(w * margin_ratio)

        x_start = max(0, margin_w)
        x_end = min(w, w - margin_w)
        y_start = max(0, margin_h)
        y_end = min(h, h - margin_h)

        centered = face_roi[y_start:y_end, x_start:x_end]

        return cv2.resize(centered, self.target_size)


def get_train_transforms(target_size=(48, 48)):
    """
    获取训练集的数据增强变换

    Args:
        target_size: 目标图像大小

    Returns:
        torchvision transforms组合
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(target_size=(48, 48)):
    """
    获取验证集的变换（仅归一化）

    Args:
        target_size: 目标图像大小

    Returns:
        torchvision transforms组合
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess_face_for_model(face_roi, target_size=(48, 48)):
    """
    为模型输入预处理人脸图像

    Args:
        face_roi: 人脸区域图像
        target_size: 目标大小

    Returns:
        预处理后的tensor
    """
    # 转换为RGB
    if len(face_roi.shape) == 2:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    elif face_roi.shape[2] == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    # 调整大小
    face_resized = cv2.resize(face_roi, target_size)

    # 应用变换
    transform = get_val_transforms(target_size)
    tensor = transform(face_resized)

    return tensor.unsqueeze(0)  # 添加batch维度


# 导出常用的函数和类
__all__ = [
    'ImagePreprocessor',
    'DataAugmentation',
    'FaceAlignment',
    'get_train_transforms',
    'get_val_transforms',
    'preprocess_face_for_model'
]