# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


# aug functions
def identity_func(img):
    return img


def autocontrast_func(img, cutoff=0):
    """
    same output as PIL.ImageOps.autocontrast
    """
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    """
    same output as PIL.ImageOps.equalize
    PIL's implementation is different from cv2.equalize
    """
    n_bins = 256

    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0:
            return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    """
    like PIL, rotate by degree, not radians
    """
    h, w = img.shape[0], img.shape[1]
    center = w / 2, h / 2
    m = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, m, (w, h), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    """
    same output as PIL.ImageOps.posterize
    """
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    """
    same output as PIL.ImageEnhance.Color
    """
    #  implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    m = np.float32(
        [[0.886, -0.114, -0.114], [-0.587, 0.413, -0.587], [-0.299, -0.299, 0.701]]
    ) * factor + np.float32([[0.114], [0.587], [0.299]])
    out = np.matmul(img, m).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
    same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = (
        np.array([(el - mean) * factor + mean for el in range(256)])
        .clip(0, 255)
        .astype(np.uint8)
    )
    out = table[img]
    return out


def brightness_func(img, factor):
    """
    same output as PIL.ImageEnhance.Contrast
    """
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    """
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    """
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    h, w = img.shape[0], img.shape[1]
    m = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(
        img, m, (w, h), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    """
    same output as PIL.Image.transform
    """
    h, w = img.shape[0], img.shape[1]
    m = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(
        img, m, (w, h), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    """
    same output as PIL.Image.transform
    """
    h, w = img.shape[0], img.shape[1]
    m = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(
        img, m, (w, h), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def posterize_func(img, bits):
    """
    same output as PIL.ImageOps.posterize
    """
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    h, w = img.shape[0], img.shape[1]
    m = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(
        img, m, (w, h), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * h), int(rw * w)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, h)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, w)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


# level to args
def enhance_level_to_args(max_level):
    def level_to_args(level):
        return ((level / max_level) * 1.8 + 0.1,)

    return level_to_args


def shear_level_to_args(max_level, replace_value):
    def level_to_args(level):
        level = (level / max_level) * 0.3
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def translate_level_to_args(translate_const, max_level, replace_value):
    def level_to_args(level):
        level = (level / max_level) * float(translate_const)
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def cutout_level_to_args(cutout_const, max_level, replace_value):
    def level_to_args(level):
        level = int((level / max_level) * cutout_const)
        return (level, replace_value)

    return level_to_args


def solarize_level_to_args(max_level):
    def level_to_args(level):
        level = int((level / max_level) * 256)
        return (level,)

    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(max_level):
    def level_to_args(level):
        level = int((level / max_level) * 4)
        return (level,)

    return level_to_args


def rotate_level_to_args(max_level, replace_value):
    def level_to_args(level):
        level = (level / max_level) * 30
        if np.random.random() < 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


func_dict = {
    "Identity": identity_func,
    "AutoContrast": autocontrast_func,
    "Equalize": equalize_func,
    "Rotate": rotate_func,
    "Solarize": solarize_func,
    "Color": color_func,
    "Contrast": contrast_func,
    "Brightness": brightness_func,
    "Sharpness": sharpness_func,
    "ShearX": shear_x_func,
    "TranslateX": translate_x_func,
    "TranslateY": translate_y_func,
    "Posterize": posterize_func,
    "ShearY": shear_y_func,
}

translate_const = 10
max_level = 10
replace_value = (128, 128, 128)
arg_dict = {
    "Identity": none_level_to_args,
    "AutoContrast": none_level_to_args,
    "Equalize": none_level_to_args,
    "Rotate": rotate_level_to_args(max_level, replace_value),
    "Solarize": solarize_level_to_args(max_level),
    "Color": enhance_level_to_args(max_level),
    "Contrast": enhance_level_to_args(max_level),
    "Brightness": enhance_level_to_args(max_level),
    "Sharpness": enhance_level_to_args(max_level),
    "ShearX": shear_level_to_args(max_level, replace_value),
    "TranslateX": translate_level_to_args(translate_const, max_level, replace_value),
    "TranslateY": translate_level_to_args(translate_const, max_level, replace_value),
    "Posterize": posterize_level_to_args(max_level),
    "ShearY": shear_level_to_args(max_level, replace_value),
}


class RandomAugment(object):
    def __init__(self, n=2, m=10, is_pil=False, augs=None):
        self.n = n
        self.m = m
        self.is_pil = is_pil
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.augs, self.n)
        return [(op, 0.5, self.m) for op in sampled_ops]

    def __call__(self, img):
        if self.is_pil:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img


if __name__ == "__main__":
    a = RandomAugment()
    img = np.random.randn(32, 32, 3)
    a(img)
