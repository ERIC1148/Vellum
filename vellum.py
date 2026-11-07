#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vellum · 手机照片 → 扫描件，全在本机。

pipeline：
  detect   多路候选找页面四边形（亮度/色度掩码 + Lab Canny），按边缘 ΔE × 面积打分
  refine   回到原图分辨率，沿法线找颜色跳变，亚像素 + 稳健直线拟合
  rectify  Zhang–He 闭式解估计真实宽高比，按内容吸附 A4 / Letter / 16:9
  flatten  文字行 + 横线拟合二维位移场（整页多项式 + 逐行斜率网格），拉平行、展平书页
  enhance  去阴影 / 白点估计 / 墨色加深 / 填实空心笔画 / 去污点 / 去残边
  ocr      macOS Vision，叠一层透明文字，PDF 可搜索、可复制
  export   手写 PDF writer：JPEG 原样嵌入，只插值一次

用法：vellum -h，详见同目录 README.md。
"""
from __future__ import annotations

import argparse
import copy
import datetime
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = "1.3.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp"}
MODES = ("auto", "doc", "color", "notes", "photo", "slide", "bw", "none")
MODE_NAMES = {
    "doc": "doc 文档",
    "color": "color 彩色",
    "photo": "photo 保真",
    "slide": "slide 幻灯片",
    "bw": "bw 黑白",
    "none": "none 原色",
    "notes": "notes 笔记",
}
# 常见纸张/屏幕的宽高比（宽/高，竖向）。估计出的比例和它们相差 2.5% 以内就对齐过去。
STANDARD_RATIOS = {
    "A4": 1 / math.sqrt(2),
    "Letter": 8.5 / 11,
    "4:3": 3 / 4,
    "16:9": 9 / 16,
    "16:10": 10 / 16,
    "证件卡": 54 / 85.6,
    "3:2": 2 / 3,
}
A4_PT = (595.276, 841.890)
LETTER_PT = (612.0, 792.0)


class ScanError(Exception):
    pass


# ───────────────────────────── 读图 ─────────────────────────────

def load_image(path: str) -> np.ndarray:
    """读成 BGR uint8，按 EXIF 方向转正。HEIC 等 Pillow 不认识的格式交给 macOS 自带的 sips。"""
    im = None
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        im.load()
    except Exception:
        im = None
    if im is None:
        tmpd = tempfile.mkdtemp(prefix="vellum-")
        try:
            out = os.path.join(tmpd, "in.jpg")
            r = subprocess.run(
                ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100", path, "--out", out],
                capture_output=True,
            )
            if r.returncode != 0 or not os.path.exists(out):
                raise ScanError(f"无法读取图片：{path}")
            im = ImageOps.exif_transpose(Image.open(out))
            im.load()
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    im = im.convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


# ─────────────────────────── 几何工具 ───────────────────────────

def order_quad(pts) -> np.ndarray:
    """排成 左上、右上、右下、左下（图像坐标，y 向下即顺时针）。"""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def quad_area(q) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def quad_ok(q, w, h) -> bool:
    if quad_area(q) < 0.08 * w * h:
        return False
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if not 35 <= math.degrees(math.acos(float(np.clip(cosang, -1, 1)))) <= 145:
            return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) >= 0.12 * min(w, h)


def contour_to_quad(c):
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_quad(ap)
        if len(ap) < 4:
            break
    # 角被遮挡或卷起时多边形近似不成四边形：在凸包顶点里挑面积最大的四个点
    pts = cv2.approxPolyDP(hull, 0.005 * peri, True).reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return None
    if len(pts) > 24:
        pts = pts[np.linspace(0, len(pts) - 1, 24).astype(int)]
    idx = np.array(list(itertools.combinations(range(len(pts)), 4)))
    P = pts[idx]
    x, y = P[..., 0], P[..., 1]
    areas = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1))
    return order_quad(P[int(areas.argmax())])


def line_intersect(l1, l2):
    (p, v), (q, u) = l1, l2
    den = v[0] * u[1] - v[1] * u[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * u[1] - (q[1] - p[1]) * u[0]) / den
    return p + t * v


def outward_normal(p0, p1, center):
    d = p1 - p0
    n = np.array([d[1], -d[0]], np.float64) / (np.linalg.norm(d) + 1e-9)
    return n if np.dot(n, (p0 + p1) / 2 - center) >= 0 else -n


# ─────────────────────────── 页面检测 ───────────────────────────

def edge_contrast(lab, q, off):
    """沿四条边内外各取样，比较 Lab 颜色差。贴着图片边框的边不计。返回 (对比度, 有效边数)。"""
    h, w = lab.shape[:2]
    c = q.mean(0)
    scores = []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        if all(min(p[0], p[1], w - 1 - p[0], h - 1 - p[1]) < 4 for p in (p0, p1)):
            continue
        n = outward_normal(p0, p1, c)
        t = np.linspace(0.1, 0.9, 40)[:, None]
        base = p0 + t * (p1 - p0)
        pin, pout = base - n * off, base + n * off
        ok = (
            (pout[:, 0] >= 0) & (pout[:, 0] <= w - 1) & (pout[:, 1] >= 0) & (pout[:, 1] <= h - 1)
            & (pin[:, 0] >= 0) & (pin[:, 0] <= w - 1) & (pin[:, 1] >= 0) & (pin[:, 1] <= h - 1)
        )
        if ok.sum() < 10:
            continue
        a = lab[pin[ok, 1].round().astype(int), pin[ok, 0].round().astype(int)]
        b = lab[pout[ok, 1].round().astype(int), pout[ok, 0].round().astype(int)]
        scores.append(float(np.median(np.linalg.norm(a - b, axis=1))) / 100.0)
    return (float(np.mean(scores)) if scores else 0.0), len(scores)


def detect_page(img):
    """在缩小图上找页面四边形。返回 (原图坐标的四边形 或 None, 置信度)。"""
    H, W = img.shape[:2]
    s = min(1.0, 900.0 / max(H, W))
    small = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    lab8 = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    lab = lab8.astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    k = max(3, int(round(min(h, w) * 0.02)) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    contours = []

    # 1) 区域法：纸通常比背景亮、颜色淡
    bright_neutral = cv2.normalize(lab[..., 0] * 2.55 - 2.0 * chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    region_masks = []
    for src in (bright_neutral, lab8[..., 0]):
        _, m = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        region_masks.append(m)
    region_masks.append(255 - region_masks[-1])  # 白桌面上的深色文件
    # 和照片四周（多半是背景）颜色差得远的区域：彩色证书、深色幻灯片也适用
    bw = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, 0), axis=2)
    dist8 = np.clip(dist * 2.55, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region_masks.append(m)
    for m in region_masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += sorted(cs, key=cv2.contourArea, reverse=True)[:4]

    # 2) 边缘法：几组 Canny 阈值，亮度和两个颜色通道都找，闭合轮廓
    # 固定增益放大色度（不用 min-max 拉伸，否则灰色照片的色噪会被放大成满屏边缘）
    a8 = np.clip((lab8[..., 1].astype(np.float32) - 128) * 3 + 128, 0, 255).astype(np.uint8)
    b8 = np.clip((lab8[..., 2].astype(np.float32) - 128) * 3 + 128, 0, 255).astype(np.uint8)
    for lo, hi in ((20, 60), (40, 120), (75, 200)):
        e = cv2.Canny(gray, lo, hi)
        if lo == 40:
            e |= cv2.Canny(a8, lo, hi) | cv2.Canny(b8, lo, hi)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
        cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) > 0.08 * w * h]
        contours += sorted(cs, key=cv2.contourArea, reverse=True)[:8]

    off = max(3.0, 0.006 * max(h, w))
    best, best_score, best_contrast = None, 0.0, 0.0
    for c in contours:
        if cv2.contourArea(c) < 0.08 * w * h:
            continue
        q = contour_to_quad(c)
        if q is None or not quad_ok(q, w, h):
            continue
        contrast, nsides = edge_contrast(lab, q, off)
        if nsides == 0:
            continue
        frac = quad_area(q) / (w * h)
        # 对比度够清楚（ΔE≥25）就不再加分，让面积说了算：证书外缘 vs 内侧花边、整页 vs 页内表格，都取外面那个
        score = min(1.0, contrast / 0.25) * math.sqrt(frac) * (nsides / 4.0) ** 0.25
        if score > best_score:
            best, best_score, best_contrast = q, score, contrast
    if best is None or best_contrast < 0.06:
        return None, best_contrast
    return best / s, best_contrast


def refine_quad(img, q):
    """在原图分辨率上沿每条边的法线找颜色跳变最强的位置，用稳健直线拟合出精确的边，再求交点。

    用 Lab 颜色差而不只是亮度（蓝色幻灯片对黑色边框亮度差很小）；离粗略位置越远权重越低，
    避免跳到证书内侧的花边、表格线上。
    """
    H, W = img.shape[:2]
    m = min(H, W)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), max(1.0, m / 900.0)), cv2.COLOR_BGR2LAB).astype(np.float32)
    chans = [lab[..., 0], lab[..., 1] * 1.5, lab[..., 2] * 1.5]  # 颜色差加权，和亮度差可比
    r = max(6, int(0.015 * m))
    us = np.arange(-r, r + 1, dtype=np.float64)
    prior = np.exp(-0.5 * (us[1:-1] / (r / 2.5)) ** 2)
    center = q.mean(0)
    lines = []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        d = p1 - p0
        tdir = d / (np.linalg.norm(d) + 1e-9)
        n = outward_normal(p0, p1, center)
        line = (p0.copy(), tdir)
        ts = np.linspace(0.06, 0.94, 80)
        base = p0[None, :] + ts[:, None] * d[None, :]
        P = base[:, None, :] + us[None, :, None] * n[None, None, :]
        inb = ((P[..., 0] >= 1) & (P[..., 0] <= W - 2) & (P[..., 1] >= 1) & (P[..., 1] <= H - 2)).all(1)
        if inb.sum() >= 12:
            mx, my = P[inb, :, 0].astype(np.float32), P[inb, :, 1].astype(np.float32)
            vals = np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                             for c in chans], -1).astype(np.float64)
            base_ok = base[inb]
            grad = np.linalg.norm(vals[:, :-2] - vals[:, 2:], axis=-1) * prior[None, :]
            k = grad.argmax(1)
            st = grad[np.arange(len(k)), k]
            # 抛物线插值到亚像素
            km, kp = np.clip(k - 1, 0, grad.shape[1] - 1), np.clip(k + 1, 0, grad.shape[1] - 1)
            gm, gp = grad[np.arange(len(k)), km], grad[np.arange(len(k)), kp]
            den = gm - 2 * st + gp
            sub = np.where(np.abs(den) > 1e-6, 0.5 * (gm - gp) / den, 0.0).clip(-0.5, 0.5)
            pos = us[k + 1] + sub
            good = st > max(3.0, 0.35 * float(np.median(st)))
            pts = (base_ok + pos[:, None] * n)[good]
            if len(pts) >= 10:
                for _ in range(4):
                    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                    nn = np.array([-vy, vx], np.float64)
                    res = np.abs((pts - [x0, y0]) @ nn)
                    keep = res < max(1.5, 2.5 * float(np.median(res)))
                    if keep.all() or keep.sum() < 10:
                        break
                    pts = pts[keep]
                line = (np.array([x0, y0], np.float64), np.array([vx, vy], np.float64))
        lines.append(line)
    new = []
    for i in range(4):
        p = line_intersect(lines[i - 1], lines[i])
        if p is None or np.linalg.norm(p - q[i]) > 4 * r:
            p = q[i]
        new.append(p)
    new = order_quad(new)
    return new if quad_ok(new, W, H) else q


def estimate_ratio(q, W, H):
    """估计页面真实的宽/高。用 Zhang & He（微软 Whiteboard It）的闭式解，从四边形反推相机焦距再求比例；
    不可靠时退回到对边平均长度之比。"""
    estimate_ratio.last_f = None
    tl, tr, br, bl = q
    naive = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / (
        np.linalg.norm(bl - tl) + np.linalg.norm(br - tr) + 1e-9)
    m1, m2, m3, m4 = (np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br))
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    except FloatingPointError:
        return naive, "naive"
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    u0, v0 = W / 2.0, H / 2.0
    if abs(n2[2]) < 1e-4 or abs(n3[2]) < 1e-4:
        r = math.sqrt((n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2 + 1e-12))
        return (r, "persp") if 0.75 < r / naive < 1.33 else (naive, "naive")
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
           + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0:
        return naive, "naive"
    f = math.sqrt(f2)
    estimate_ratio.last_f = f
    if not 0.3 * max(W, H) <= f <= 6 * max(W, H):
        return naive, "naive"
    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
    Ai = np.linalg.inv(A)
    B = Ai.T @ Ai
    r = math.sqrt(float(n2 @ B @ n2) / float(n3 @ B @ n3))
    if not 0.7 < r / naive < 1.4:
        return naive, "naive"
    return r, "persp"


# 不同内容只对齐到它可能的比例：纸质文档不会是 16:9，幻灯片不会是 A4
SNAP_SETS = {
    "doc": ("A4", "Letter"),
    "color": ("A4", "Letter"),
    "bw": ("A4", "Letter"),
    "photo": ("A4", "Letter", "证件卡", "3:2", "4:3", "16:9"),
    "slide": ("16:9", "16:10", "4:3"),
    "none": ("A4", "Letter", "证件卡"),
    "notes": ("A4", "Letter"),
}


def snap_ratio(r, mode="none"):
    names = SNAP_SETS.get(mode, SNAP_SETS["none"])
    for name, c in ((n, STANDARD_RATIOS[n]) for n in names):
        for cand, suffix in ((c, ""), (1 / c, " 横向")):
            if abs(r / cand - 1) < 0.025:
                return cand, name + suffix
    return r, None


def parse_ratio(s):
    s = s.strip().lower()
    if s in ("auto", "free", "raw"):
        return s
    for name, c in STANDARD_RATIOS.items():
        if s == name.lower():
            return c
    m = re.fullmatch(r"([\d.]+)\s*[:x/]\s*([\d.]+)", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    raise argparse.ArgumentTypeError(f"看不懂的比例：{s}（可用 auto、free、A4、Letter、16:9、宽:高）")


def page_geometry(q, ratio, scale):
    """矫正后页面的尺寸、内缩量和透视矩阵（原图 → 页面）。"""
    tl, tr, br, bl = q
    wpx = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hpx = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if ratio:
        if wpx / hpx > ratio:
            w, h = wpx, wpx / ratio
        else:
            w, h = hpx * ratio, hpx
    else:
        w, h = wpx, hpx
    w, h = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    inset = max(2, int(round(0.003 * min(w, h))))  # 去掉边缘过渡像素
    return w, h, inset, M


def warp_page(img, q, ratio, scale, field=None):
    """透视矫正。有展平场 / 书页模型时，把“展平 + 透视”合成一次重映射，原图只插值一次。"""
    w, h, inset, M = page_geometry(q, ratio, scale)
    if field is None:
        out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return out[inset:h - inset, inset:w - inset]
    outside = None
    if field.get("kind") == "book":
        X, Ys = book_maps(field, w, h)
        X, Ys = X[inset:-inset, :], Ys[inset:-inset, :]
        X, Ys = X[:, inset:-inset], Ys[:, inset:-inset]
        # 展开后的平面比矫正页大一圈：落到页面外（桌面）的像素当纸面白
        m = 0.004 * min(w, h)
        outside = (X < m) | (X > w - 1 - m) | (Ys < m) | (Ys > h - 1 - m)
    else:
        ow, oh = w - 2 * inset, h - 2 * inset
        X, Y = np.meshgrid(np.arange(ow, dtype=np.float32) + inset, np.arange(oh, dtype=np.float32) + inset)
        Ys = Y + field_displacement(field, X, Y, w, h)
    Mi = np.linalg.inv(M).astype(np.float32)
    den = Mi[2, 0] * X + Mi[2, 1] * Ys + Mi[2, 2]
    mx = (Mi[0, 0] * X + Mi[0, 1] * Ys + Mi[0, 2]) / den
    my = (Mi[1, 0] * X + Mi[1, 1] * Ys + Mi[1, 2]) / den
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if outside is not None and outside.any():
        paper = np.median(out[~outside].reshape(-1, out.shape[-1]) if out.ndim == 3 else out[~outside], axis=0)
        out[cv2.dilate(outside.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = paper
    return out


# ─────────────────────── 文字行拉平 / 书页展平 ───────────────────────
# 模型：观测到的文字行 y = 该行的真实高度 t_i + d(x, y)。
# d 是 x、y 的二维多项式（x 的 1–4 次 × y 的 0–2 次，共 12 项，不含纯 y 项——纯 y 项会被各行的 t_i 吸收）。
# 能表达：整页倾斜/手写行斜（x）、行斜度随高度变化（xy）、书页弯曲（x²、x³、x⁴）、上下弯曲程度不同（x²y、x²y²…）。
# 思路和 page_dewarp 的“三次曲面”相近，但只在二维图像上做线性最小二乘，不需要 3D 优化，几十毫秒。
FIELD_TERMS = [(i, j) for j in range(3) for i in range(1, 5)]


def field_terms(nlines):
    """行少就只用低阶项，防止过拟合：4–7 行只拉平倾斜，8–14 行再加弯曲，15 行以上用全部 12 项。"""
    if nlines >= 15:
        return FIELD_TERMS
    if nlines >= 8:
        return [(1, 0), (2, 0), (1, 1), (2, 1), (3, 0)]
    return [(1, 0), (1, 1)]


def field_displacement(field, X, Y, w, h):
    """field = {terms, coef, xr, yr}。超出文字行覆盖范围的地方按边界值延伸，不外推多项式。"""
    xn = np.clip((X - w / 2.0) / (w / 2.0), *field["xr"]).astype(np.float32)
    yn = np.clip((Y - h / 2.0) / (h / 2.0), *field["yr"]).astype(np.float32)
    d = np.zeros_like(xn)
    for (i, j), c in zip(field["terms"], field["coef"]):
        d += np.float32(c) * xn ** i * yn ** j
    G = field.get("grid")
    if G is not None:  # 第二级：逐行斜率修正，存在粗网格上，双线性取样
        gh, gw = G.shape
        mx = (np.asarray(X, np.float32) / np.float32(w)) * gw - 0.5
        my = (np.asarray(Y, np.float32) / np.float32(h)) * gh - 0.5
        d += cv2.remap(G, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return d * np.float32(h / 2.0)


def apply_field(page, field):
    h, w = page.shape[:2]
    if field.get("kind") == "book":
        X, Ys = book_maps(field, w, h)
        out = cv2.remap(page, X, Ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        outside = (X < 0) | (X > w - 1) | (Ys < 0) | (Ys > h - 1)
        if outside.any():
            out[outside] = np.median(out[~outside].reshape(-1, out.shape[-1]), axis=0)
        return out
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(page, X, Y + field_displacement(field, X, Y, w, h), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def find_text_lines(page):
    """找文字行，返回 ([(xs, ys), ...] 每段文字行中心线的采样点，页面像素坐标), 字高（像素）。"""
    h, w = page.shape[:2]
    s = min(1.0, 1600.0 / max(h, w))
    sm = cv2.resize(page, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    f = sm.astype(np.float32)
    lum = luminance(f[..., 0], f[..., 1], f[..., 2])
    n = lum / estimate_background([lum], lum)[0]
    ink = (n < 0.72).astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num < 30:
        return [], 0.0
    cw, chh, area = st[1:, 2], st[1:, 3], st[1:, 4]
    ok = area >= 6
    if ok.sum() < 30:
        return [], 0.0
    ch = max(4.0, float(np.percentile(chh[ok], 75)))  # 手写笔画会断成碎块，中位数偏小；取 75 分位接近字高
    keep = np.zeros(num, bool)
    keep[1:] = ok & (chh <= 3 * ch) & (cw <= 0.5 * sm.shape[1])  # 去掉 logo、插图、表格竖线
    mask = keep[lab].astype(np.uint8)
    kx, ky = max(3, int(round(2.0 * ch))), max(1, int(round(0.25 * ch)))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    ys, xs = np.nonzero(mask)
    ids = lab2[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, xs, ys = ids[order], xs[order], ys[order]
    starts = np.searchsorted(ids, np.arange(n2 + 1))
    bin_w = max(2, int(round(0.8 * ch)))
    lines = []
    for i in range(1, n2):
        x0, y0, ww, hh = st2[i, :4]
        # 一整行或一个词都行：词也带着局部斜率信息（每段有自己的高度偏移，只贡献“斜不斜、弯不弯”）
        if ww < 2.5 * ch or hh > 3.5 * ch or ww < 1.5 * hh:
            continue
        lx, ly = xs[starts[i]:starts[i + 1]], ys[starts[i]:starts[i + 1]]
        if lx.size < 20:
            continue
        b = (lx - x0) // bin_w
        cnt = np.bincount(b)
        good = cnt >= 3
        if good.sum() < 4:
            continue
        cx = np.bincount(b, lx)[good] / cnt[good]
        cy = np.bincount(b, ly)[good] / cnt[good]
        lines.append((cx / s, cy / s))
    # 长横线（表格线、下划线、答题横线）是最好的“直线参照”：横向开运算去掉竖线和文字，只留横线
    # 先竖向加厚几个像素：弯曲/倾斜的细线是“台阶”，不加厚的话横向开运算会把它整条吃掉
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(0.2 * ch)) | 1)))
    horiz = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(1.5 * ch)), 1)))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for i in range(1, n3):
        x0, y0, ww, hh, area = st3[i]
        if ww < 8 * ch or area / ww > 0.6 * ch:
            continue
        ry, rx = np.nonzero((lab3[y0:y0 + hh, x0:x0 + ww] == i) & (ink[y0:y0 + hh, x0:x0 + ww] > 0))
        b = rx // bin_w
        cnt = np.bincount(b)
        good = cnt >= 1
        cx = (np.bincount(b, rx)[good] / cnt[good]) + x0
        cy = (np.bincount(b, ry)[good] / cnt[good]) + y0
        if good.sum() >= 6:
            lines.append((cx / s, cy / s))
    return lines, ch / s


def line_grid(lines, X, Y, L, r, wgt, full_w, full_h, ch, px):
    """逐行斜率修正网格（归一化位移单位）。只用够长的段；斜率限制在 ±4°。"""
    cell = max(4.0, ch / 2.5)
    gw, gh = max(8, int(full_w / cell)), max(8, int(full_h / cell))
    gxs = (np.arange(gw) + 0.5) * full_w / gw
    gys = (np.arange(gh) + 0.5) * full_h / gh
    num = np.zeros((gh, gw), np.float64)
    den = np.zeros((gh, gw), np.float64)
    used = 0
    rpx = r * px
    for k in range(len(lines)):
        m = L == k
        x, y, rr, ww = X[m], Y[m], rpx[m], wgt[m]
        if x.size < 6 or np.ptp(x) < 4 * ch:
            continue
        xm = np.average(x, weights=ww)
        a = np.sum(ww * (x - xm) * rr) / max(np.sum(ww * (x - xm) ** 2), 1e-9)
        a = float(np.clip(a, -0.07, 0.07))  # ±4°
        if abs(a) < math.tan(math.radians(0.25)):
            continue
        yc, x0, x1 = float(np.average(y, weights=ww)), float(x.min()), float(x.max())
        wy = np.exp(-0.5 * ((gys - yc) / (0.9 * ch)) ** 2)[:, None]
        dx = np.maximum(0, np.maximum(x0 - gxs, gxs - x1))
        wx = np.exp(-0.5 * (dx / (1.5 * ch)) ** 2)[None, :]
        wgrid = wy * wx * min(1.0, np.ptp(x) / (10 * ch))
        num += wgrid * (a * (np.clip(gxs, x0 - 2 * ch, x1 + 2 * ch) - xm))[None, :]
        den += wgrid
        used += 1
    if used == 0:
        return None, 0
    G = num / (den + 0.15)  # 离所有文字行都远的地方趋向 0
    G = cv2.GaussianBlur(G, (0, 0), 0.8)
    return (G / px).astype(np.float32), used


# ─────────────────────── 书页：3D 三次曲面 + 非线性优化 ───────────────────────
# 思路同 page_dewarp：纸面沿 x 方向是一条三次曲线 z(u) = (α+β)u³ − (2α+β)u² + αu（u∈[0,1]，两端 z=0），
# 针孔相机看过去；每条文字行一个高度 Y_i，每个采样点一个 x 位置 X_k，加上 3 个旋转角，
# 用 scipy least_squares（稀疏雅可比，soft-L1 稳健损失）最小化重投影误差。
# 渲染时按曲面弧长取样：书脊附近被“压窄”的字会被拉回原宽——这是二维位移场做不到的。
# 坐标全部按页宽归一化，所以模型和输出分辨率无关。

def book_z(u, a, b):
    return (a + b) * u ** 3 - (2 * a + b) * u ** 2 + a * u


def fit_book(lines, full_w, full_h, off, ch, f_page):
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except Exception:
        return None, {"error": "没有 SciPy"}
    Wn = float(full_w)
    xs, ys, L = [], [], []
    for xs_l, ys_l in lines:
        if xs_l.size < 6 or np.ptp(xs_l) < 6 * ch:
            continue
        sel = np.linspace(0, xs_l.size - 1, min(24, xs_l.size)).astype(int)
        xs.append(xs_l[sel])
        ys.append(ys_l[sel])
        L.append(np.full(sel.size, len(L)))
    n = len(xs)
    if n < 6:
        return None, {"lines": n}
    xo = (np.concatenate(xs) + off - full_w / 2.0) / Wn
    yo = (np.concatenate(ys) + off - full_h / 2.0) / Wn
    L = np.concatenate(L)
    N = xo.size
    Dn = f_page / Wn
    Y0 = np.array([yo[L == i].mean() for i in range(n)])

    def project(p):
        R = cv2.Rodrigues(p[0:3].reshape(3, 1))[0]
        a, b = p[3], p[4]
        Yi, Xk = p[5:5 + n], p[5 + n:]
        Z = book_z(Xk + 0.5, a, b)
        P = np.stack([Xk, Yi[L], -Z], 1).dot(R.T)
        dep = P[:, 2] + Dn
        return Dn * P[:, 0] / dep, Dn * P[:, 1] / dep

    def fun(p):
        x, y = project(p)
        return np.concatenate([x - xo, y - yo])

    S = lil_matrix((2 * N, 5 + n + N), dtype=np.int8)
    S[:, 0:5] = 1
    rows = np.arange(N)
    S[rows, 5 + L] = 1
    S[N + rows, 5 + L] = 1
    S[rows, 5 + n + rows] = 1
    S[N + rows, 5 + n + rows] = 1
    p0 = np.concatenate([np.zeros(5), Y0, xo])
    lo = np.concatenate([[-0.5, -0.5, -0.5, -2.0, -2.0], np.full(n + N, -np.inf)])
    hi = np.concatenate([[0.5, 0.5, 0.5, 2.0, 2.0], np.full(n + N, np.inf)])
    with np.errstate(all="ignore"):
        res = least_squares(fun, p0, jac_sparsity=S, method="trf", x_scale="jac", bounds=(lo, hi),
                            loss="soft_l1", f_scale=max(0.5, 0.15 * ch) / Wn, max_nfev=400)
    p = res.x
    before = float(np.sqrt(np.mean((yo - Y0[L]) ** 2))) * Wn
    after = float(np.sqrt(np.mean(fun(p)[N:] ** 2))) * Wn
    u = np.linspace(0, 1, 2001)
    dz = np.gradient(book_z(u, p[3], p[4]), u)
    trap = getattr(np, "trapezoid", None) or np.trapz
    arc = float(trap(np.sqrt(1 + dz ** 2), u))
    model = {"kind": "book", "rvec": p[0:3].tolist(), "a": float(p[3]), "b": float(p[4]), "Dn": Dn, "arc": arc}
    stats = {"lines": n, "before": before, "after": after, "arc": arc, "a": float(p[3]), "b": float(p[4])}
    ok = res.success and after < 0.75 * before and before - after > 0.5 and arc < 1.6
    return (model if ok else None), stats


def book_maps(model, w, h):
    """输出像素（沿曲面弧长展开的平面）→ 矫正后页面上的取样坐标。输出宽度按弧长加宽。"""
    a, b, Dn = model["a"], model["b"], model["Dn"]
    R = cv2.Rodrigues(np.array(model["rvec"], np.float64).reshape(3, 1))[0]
    u = np.linspace(0, 1, 4001)
    z = book_z(u, a, b)
    s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(u) ** 2 + np.diff(z) ** 2))])
    ow = int(round(w * s[-1]))
    cols = (np.arange(ow) + 0.5) / ow * s[-1]
    uc = np.interp(cols, s, u)
    Xc, Zc = uc - 0.5, book_z(uc, a, b)
    Yr = (np.arange(h) - h / 2.0) / w
    X = np.broadcast_to(Xc[None, :], (h, ow))
    Z = np.broadcast_to(Zc[None, :], (h, ow))
    Y = np.broadcast_to(Yr[:, None], (h, ow))
    Px = R[0, 0] * X + R[0, 1] * Y - R[0, 2] * Z
    Py = R[1, 0] * X + R[1, 1] * Y - R[1, 2] * Z
    Pz = R[2, 0] * X + R[2, 1] * Y - R[2, 2] * Z + Dn
    mx = (w / 2.0 + Dn * Px / Pz * w).astype(np.float32)
    my = (h / 2.0 + Dn * Py / Pz * w).astype(np.float32)
    return mx, my


def fit_line_field(lines, full_w, full_h, off, ch):
    """用所有文字行拟合位移场。返回 (系数 或 None, 统计)。只有确实能让行更直、且位移幅度合理时才采用。"""
    if len(lines) < 4:
        return None, {"lines": len(lines)}
    X = np.concatenate([l[0] for l in lines]) + off
    Y = np.concatenate([l[1] for l in lines]) + off
    L = np.concatenate([np.full(len(l[0]), k) for k, l in enumerate(lines)])
    if X.size < 40:
        return None, {"lines": len(lines)}
    xn = (X - full_w / 2.0) / (full_w / 2.0)
    yn = (Y - full_h / 2.0) / (full_h / 2.0)
    terms = field_terms(len(lines))
    B = np.stack([xn ** i * yn ** j for i, j in terms], 1)
    E = np.zeros((X.size, len(lines)))
    E[np.arange(X.size), L] = 1.0
    A = np.hstack([B, E])
    nt = B.shape[1]
    reg = np.r_[np.full(nt, 1e-3), np.full(len(lines), 1e-9)]
    wgt = np.ones(X.size)
    with np.errstate(all="ignore"):  # macOS Accelerate 的 matmul 偶尔误报浮点警告
        for _ in range(8):  # 迭代重加权，压住公式、分数线、连笔这类离群点
            Aw = A * wgt[:, None]
            sol = np.linalg.solve(Aw.T @ A + np.diag(reg) * X.size, Aw.T @ yn)
            r = yn - A @ sol
            sig = 1.4826 * float(np.median(np.abs(r))) + 1e-6
            wgt = 1.0 / (1.0 + (r / (2.5 * sig)) ** 2)
    if not np.all(np.isfinite(sol)):
        return None, {"lines": len(lines)}
    coef = sol[:nt]
    xr = (float(xn.min()), float(xn.max()))
    yr = (float(yn.min()), float(yn.max()))
    field = {"terms": terms, "coef": coef, "xr": xr, "yr": yr}
    px = full_h / 2.0
    # 拟合前：每行只减去自己的平均高度
    means = np.bincount(L, yn * wgt) / np.maximum(np.bincount(L, wgt), 1e-9)
    r0 = yn - means[L]
    before = float(np.sqrt(np.average(r0 ** 2, weights=wgt))) * px
    after = float(np.sqrt(np.average(r ** 2, weights=wgt))) * px
    # 第二级：每一行自己的斜率。整页多项式只能拉平“整体趋势”，手写每行斜得不一样，
    # 对多项式拟合后的残差逐段再拟一条直线，把斜率铺到粗网格上（上下按字高高斯过渡，左右在段外渐隐）。
    grid, nlev = line_grid(lines, X, Y, L, r, wgt, full_w, full_h, ch, px)
    if grid is not None:
        field["grid"] = grid
        Gs = cv2.remap(grid, (X / full_w * grid.shape[1] - 0.5).astype(np.float32).reshape(1, -1),
                       (Y / full_h * grid.shape[0] - 0.5).astype(np.float32).reshape(1, -1),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        r = r - Gs
        after = float(np.sqrt(np.average(r ** 2, weights=wgt))) * px
    gx, gy = np.meshgrid(np.linspace(0, full_w, 21), np.linspace(0, full_h, 29))
    dmax = float(np.abs(field_displacement(field, gx, gy, full_w, full_h)).max())
    stats = {"lines": len(lines), "leveled": nlev, "before": before, "after": after, "dmax": dmax,
             "explained": math.sqrt(max(0.0, before ** 2 - after ** 2))}
    # 模型解释掉的偏差要明显大于“用这么多参数去拟合噪声”能解释的量，才采用
    explained = math.sqrt(max(0.0, before ** 2 - after ** 2))
    noise_fit = 3.0 * math.sqrt(len(terms) / X.size) * after
    useful = explained > max(1.0, noise_fit) and after < 0.95 * before
    sane = dmax < max(3.0 * ch, 0.03 * full_h)  # 位移最多三个字高，再大多半是拟合坏了
    return (field if useful and sane else None), stats


def estimate_skew(gray):
    """没找到页面边缘时（已经是扫描件/截图），按文字行的水平投影估计小角度倾斜，±5°。"""
    h, w = gray.shape
    s = min(1.0, 1000.0 / max(h, w))
    sm = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(sm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    if ink.mean() < 1.0:
        return 0.0
    hh, ww = ink.shape
    c = (ww / 2, hh / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(ink, M, (ww, hh), flags=cv2.INTER_NEAREST)
        return float(np.var(r.sum(1, dtype=np.float64)))

    base = score(0.0)
    coarse = np.arange(-5, 5.01, 0.25)
    sc = [score(a) for a in coarse]
    a0 = float(coarse[int(np.argmax(sc))])
    fine = np.arange(a0 - 0.25, a0 + 0.251, 0.05)
    sf = [score(a) for a in fine]
    a = float(fine[int(np.argmax(sf))])
    if abs(a) < 0.15 or max(sf) < base * 1.05:
        return 0.0
    return a


def rotate_small(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────── 光照与增强 ───────────────────────────

def estimate_background(channels, lum, coarse=False):
    """估计纸张底色（含阴影、光照不均）。

    缩小后做形态学闭运算抹掉文字笔画；比周围亮包络暗很多的大块区域（照片、logo、色块）
    当作内容而不是阴影，用周围的底色补上，避免被“洗白”。
    """
    H, W = lum.shape
    target = 160 if coarse else 320
    f = target / float(min(H, W))
    sw, sh = max(8, int(round(W * f))), max(8, int(round(H * f)))
    k = max(3, int(round(min(sw, sh) * (0.03 if coarse else 0.016))) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lum_s = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)
    lum_bg = cv2.morphologyEx(lum_s, cv2.MORPH_CLOSE, kern)
    K = max(k * 5, int(round(min(sw, sh) * 0.12))) | 1
    env = cv2.dilate(lum_bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K)))
    env = cv2.GaussianBlur(env, (0, 0), K / 3.0)
    bad = lum_bg < 0.72 * env
    if coarse:
        # 保真模式更保守：比纸面亮度一半还暗的大片区域（深色幻灯片底、色块）一律当内容，不提亮
        paper_level = float(np.percentile(lum_bg, 90))
        bad |= lum_bg < 0.55 * paper_level
        if bad.mean() > 0.5:  # 画面以内容为主，分不清哪是光照，干脆不做光照校正
            return [np.full((H, W), max(1.0, float(np.percentile(cv2.resize(c, (sw, sh)), 90))), np.float32)
                    for c in channels]
    bad = cv2.dilate(bad.astype(np.uint8), kern)
    use_inpaint = 0 < bad.mean() < 0.6
    out = []
    for ch in channels:
        b = cv2.morphologyEx(cv2.resize(ch, (sw, sh), interpolation=cv2.INTER_AREA), cv2.MORPH_CLOSE, kern)
        if use_inpaint:
            b8 = np.clip(b, 0, 255).astype(np.uint8)
            b = cv2.inpaint(b8, bad, 3, cv2.INPAINT_TELEA).astype(np.float32)
        b = cv2.medianBlur(b.astype(np.float32), 5)
        b = cv2.GaussianBlur(b, (0, 0), max(0.8, k * 0.35))
        out.append(np.maximum(cv2.resize(b, (W, H), interpolation=cv2.INTER_CUBIC), 1.0))
    return out


def luminance(b, g, r):
    return 0.114 * b + 0.587 * g + 0.299 * r


def unsharp(x, sigma, amount):
    return x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma))


def tone_params(n):
    """n：归一化亮度，纸≈1。返回 (黑点, 白点)。白点以上全部变纯白，噪声估计自纸面。"""
    s = n[::3, ::3].ravel()
    paper = s[s > 0.8]
    if paper.size < 100:
        return 0.0, 0.92
    mp = float(np.median(paper))
    mad = float(np.median(np.abs(paper - mp))) * 1.4826
    wt = float(np.clip(mp - max(0.04, 3.5 * mad), mp - 0.15, mp - 0.03))
    ink = s[s < wt - 0.1]
    bp = float(np.percentile(ink, 2)) if ink.size > 200 else 0.0
    bp = max(0.0, min(bp, wt - 0.35))
    return bp, wt


# 墨色浓度预设：gamma 压暗中间调（笔画边缘的灰阶），target 是每片区域最深笔画至少要达到的深度（0 黑 1 白），
# bold 是笔画外扩的混合比例（约 0.5–1 像素）
INK_PRESETS = {
    0: dict(gamma=1.0, target=None, bold=0.0, fill=0.0),   # raw：只拉伸，不额外加深
    1: dict(gamma=1.3, target=0.30, bold=0.0, fill=0.5),   # normal
    2: dict(gamma=1.8, target=0.05, bold=0.0, fill=0.85),  # deep（默认）：笔芯接近纯黑
    3: dict(gamma=2.0, target=0.03, bold=0.6, fill=1.0),   # bold：再把笔画加粗一点
}
INK_NAMES = {0: "raw", 1: "normal", 2: "deep", 3: "bold"}


def apply_tone(n, bp, wt, gamma, target=None):
    """把 [黑点, 白点] 拉伸到 [0, 1] 再做 gamma。

    target 不为空时按局部笔画深度抬高黑点：一片区域里最深的笔画也很浅（笔没水、铅笔、褪色），
    就把它拉到 target 这么深；本来就深的文字几乎不变。完全空白处（局部对比度低于噪声）不处理，不会放大噪点。
    """
    if target is not None:
        h, w = n.shape
        k = max(5, int(0.016 * min(h, w)) | 1)
        f = 0.25
        lb = cv2.erode(n, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))  # 先在原分辨率取局部最深，细笔画不被平均掉
        lb = cv2.resize(lb, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        lb = cv2.GaussianBlur(lb, (0, 0), max(0.8, k * f / 4))
        lb = cv2.resize(lb, (w, h), interpolation=cv2.INTER_LINEAR)
        tt = target ** (1.0 / gamma)
        bp_local = (lb - tt * wt) / (1.0 - tt)
        c = wt - lb  # 局部笔画对比度；低于 0.07 当作纸面噪声/透背，不处理
        wgt = np.clip((c - 0.07) / 0.08, 0.0, 1.0)
        bp = bp + np.clip(bp_local - bp, 0.0, None) * wgt
        bp = np.minimum(bp, wt - 0.12)
    t = np.clip((n - bp) / (wt - bp), 0.0, 1.0)
    return t ** gamma


def despeckle(v, min_dim):
    """去掉纸面上孤立的浅色小污点（没有任何深色像素的小连通块）。i 的点、句号是深色，不受影响。"""
    nonwhite = (v < 0.93).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(nonwhite, connectivity=8)
    if n <= 1:
        return v
    amax = max(6, int((min_dim * 0.0018) ** 2))
    dark = np.bincount(lab[v < 0.6], minlength=n)
    remove = (st[:, cv2.CC_STAT_AREA] <= amax) & (dark == 0)
    remove[0] = False
    if remove.any():
        v[remove[lab]] = 1.0
    return v


def border_artifacts(v):
    """贴着边、且整块都落在边缘窄带里的暗块（桌面残边、纸边阴影、反光条），返回它们的掩码。"""
    h, w = v.shape
    band = max(6, int(0.03 * min(h, w)))
    ink = (v < 0.95).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return None
    x, y, ww, hh = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    e = 2
    touch = (x <= e) | (y <= e) | (x + ww >= w - e) | (y + hh >= h - e)
    in_band = (x + ww <= band) | (y + hh <= band) | (x >= w - band) | (y >= h - band)
    remove = touch & in_band
    remove[0] = False
    if not remove.any():
        return None
    return cv2.dilate(remove[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def sharpen_sigma(shape, scale):
    # 半径跟着原始照片的分辨率走：放大过的图，锐化半径也要跟着放大，否则只会放大噪点
    return max(0.7, min(shape) / 2600.0) * max(1.0, scale)


def fill_hollow(v, amount, scale):
    """填实“空心笔画”：圆珠笔/中性笔笔芯反光，拍出来是两边深中间浅；开运算去掉笔画里细细的亮缝。"""
    if amount <= 0:
        return v
    k = 3 if scale < 1.6 else 5
    opened = cv2.morphologyEx(v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return v * (1 - amount) + opened * amount


def embolden(v, amount, scale):
    """笔画外扩：v 越小越黑，取局部最小值就是把黑色往外扩一点，再按 amount 混合。"""
    if amount <= 0:
        return v
    k = 3 if scale < 1.6 else 5
    return v * (1 - amount) + cv2.erode(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) * amount


def enhance_doc(img, ink=2, sharpen=0.45, scale=1.0, gamma=None):
    """白底黑字：灰度，背景纯白，笔画加深，保留抗锯齿边缘（不硬二值化），所以和原图一样清晰。"""
    p = INK_PRESETS[ink]
    f = img.astype(np.float32)
    lum = luminance(f[..., 0], f[..., 1], f[..., 2])
    bg = estimate_background([lum], lum)[0]
    n = lum / bg
    n = unsharp(n, sharpen_sigma(n.shape, scale), sharpen)
    bp, wt = tone_params(n)
    v = apply_tone(n, bp, wt, p["gamma"] if gamma is None else gamma, p["target"])
    v = fill_hollow(v, p["fill"], scale)
    v = embolden(v, p["bold"], scale)
    m = border_artifacts(v)
    if m is not None:
        v[m] = 1.0
    v = despeckle(v, min(v.shape))
    return Image.fromarray((v * 255 + 0.5).astype(np.uint8))


def enhance_bw(img, ink=2, scale=1.0):
    v = np.asarray(enhance_doc(img, ink=ink, sharpen=0.8, scale=scale, gamma=1.0), np.float32) / 255.0
    return Image.fromarray(((v > 0.6) * 255).astype(np.uint8)).convert("1", dither=Image.Dither.NONE)


def enhance_color(img, ink=2, sharpen=0.5, boost=1.12, scale=1.0):
    """彩色文档：每个通道各自除以底色（同时去阴影和白平衡），纸变纯白，印章、彩笔、logo 保留颜色。"""
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = luminance(b, g, r)
    bgb, bgg, bgr = estimate_background([b, g, r], lum)
    nb, ng, nr = b / bgb, g / bgg, r / bgr
    nl = luminance(nb, ng, nr)
    delta = sharpen * (nl - cv2.GaussianBlur(nl, (0, 0), sharpen_sigma(nl.shape, scale)))
    nl2 = nl + delta
    bp, wt = tone_params(nl2)
    p = INK_PRESETS[ink]
    # 彩色文档的 gamma 收一点，压太狠颜色会发脏
    fl = apply_tone(nl2, bp, wt, 1.0 + (p["gamma"] - 1.0) * 0.75, p["target"])
    fl = embolden(fill_hollow(fl, p["fill"], scale), p["bold"], scale)
    ratio = fl / np.maximum(nl2, 1e-3)
    chans = [np.clip((c + delta) * ratio, 0, 1.2) for c in (nb, ng, nr)]
    # 增加一点饱和度
    chans = [fl + (c - fl) * boost for c in chans]
    # 接近白色、又几乎没有颜色的像素直接变纯白中性
    cmax = np.maximum(np.maximum(chans[0], chans[1]), chans[2])
    cmin = np.minimum(np.minimum(chans[0], chans[1]), chans[2])
    wgt = np.clip((0.08 - (cmax - cmin)) / 0.08, 0, 1) * np.clip((fl - 0.85) / 0.15, 0, 1)
    chans = [c * (1 - wgt) + fl * wgt for c in chans]
    out = np.clip(np.dstack(chans), 0, 1)
    v = luminance(out[..., 0], out[..., 1], out[..., 2])
    chroma = out.max(2) - out.min(2)
    m = border_artifacts(np.where(chroma < 0.12, v, 1.0))
    if m is not None:
        out[m] = 1.0
    out_rgb = np.ascontiguousarray((out[..., ::-1] * 255 + 0.5).astype(np.uint8))
    return Image.fromarray(out_rgb)


def enhance_notes(img, ink=2, colors=8, scale=1.0):
    """笔记 · 调色板压缩（思路来自 noteshrink）。

    先走彩色文档流程（纸面纯白、墨色加深、保留颜色），再把“墨”的颜色用 k-means 聚成 colors-1 种，
    加饱和、拉开深浅，纸面固定为白色。结果是索引色图：红蓝黑笔迹分得清，文件比 JPEG 小很多，边缘没有压缩噪点。
    """
    rgb = np.asarray(enhance_color(img, ink=ink, scale=scale)).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    v = rgb.max(2)
    sat = (v - rgb.min(2)) / np.maximum(v, 1e-3)
    fg = (v < 0.78) | ((sat > 0.28) & (v < 0.96))
    idx = np.zeros((h, w), np.uint8)
    pal = [255, 255, 255]
    k = max(1, min(15, int(colors) - 1))
    if fg.sum() >= 50:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)  # float：L 0–100，a/b ±127
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        colored = fg & (chroma > 18)
        neutral = fg & ~colored
        rng = np.random.default_rng(0)
        # 彩色墨（红章、蓝笔）单独聚类，哪怕只占一点点也给它留位置；其余给黑灰几级（文字边缘更平滑）
        k_col = 0
        if colored.sum() > max(200, 0.002 * fg.sum()):
            k_col = min(max(1, k // 2), k - 1) if k > 1 else 0
        k_gray = k - k_col
        cols = []

        def cluster(mask, kk, weight):
            px = lab[mask] * weight
            sub = px[rng.choice(len(px), min(30000, len(px)), replace=False)].astype(np.float32)
            kk = min(kk, len(sub))
            _, _, c = cv2.kmeans(sub, kk, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
                                 3, cv2.KMEANS_PP_CENTERS)
            d = ((px[:, None, :] - c[None, :, :]) ** 2).sum(2)
            return c / weight, d.argmin(1)

        if k_gray > 0 and neutral.any():
            c, a = cluster(neutral, k_gray, np.array([1.0, 0.5, 0.5], np.float32))
            idx[neutral] = (a + 1 + len(cols)).astype(np.uint8)
            cols += list(c)
        if k_col > 0:
            c, a = cluster(colored, k_col, np.array([0.5, 2.0, 2.0], np.float32))
            idx[colored] = (a + 1 + len(cols)).astype(np.uint8)
            cols += list(c)
        # 调色板“上色”：黑灰拉开深浅（最深接近纯黑），彩色加饱和
        cols = np.array(cols, np.float32)
        L = cols[:, 0]
        chroma_c = np.hypot(cols[:, 1], cols[:, 2])
        gray = chroma_c <= 18
        if gray.any():
            g = L[gray]
            L[gray] = 6 + (g - g.min()) / max(g.max() - g.min(), 1e-6) * 64 if gray.sum() > 1 else 6
        cols[~gray, 1:] *= 1.35
        cols[~gray, 0] = np.minimum(L[~gray], 58)
        cols[:, 0] = L if not (~gray).any() else np.where(gray, L, cols[:, 0])
        rgbc = np.clip(cv2.cvtColor(cols.reshape(1, -1, 3), cv2.COLOR_LAB2RGB)[0], 0, 1)
        pal += [int(round(c * 255)) for c in rgbc.ravel()]
    im = Image.fromarray(idx, "P")
    im.putpalette(pal)
    return im


def enhance_photo(img, strength=0.9, sharpen=0.45, vib=1.04, slide=False, wb=0.6):
    """保真：证书、奖状、演示文稿。只温和地拉平光照、白平衡、自动色阶、锐化，保留底纹和颜色。"""
    f = img.astype(np.float32)
    if slide:
        # 去摩尔纹/像素网格：先按分辨率做轻度模糊抹掉周期纹理，再用保边滤波压平色块，文字边缘靠后面的锐化找回
        sig = max(0.8, max(img.shape[:2]) / 2400.0)
        sm = cv2.GaussianBlur(img, (0, 0), sig)
        f = cv2.bilateralFilter(sm, 9, 30, 9).astype(np.float32)
        sharpen = max(sharpen, 0.9)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = luminance(b, g, r)
    bg = estimate_background([lum], lum, coarse=True)[0]
    target = float(np.percentile(bg, 90))
    gain = np.clip((target / bg) ** strength, 0.7, 2.0)
    f = f * gain[..., None]
    # 白平衡：取最亮的一成里颜色不太艳的像素（纸面、白字），只校正偏色的一部分（wb），
    # 因为分不清是灯光偏黄还是纸本来就是米黄色，全校正会把米色证书洗成白纸
    lum = luminance(f[..., 0], f[..., 1], f[..., 2])
    cmax, cmin = f.max(2), f.min(2)
    sel = (lum > np.percentile(lum, 90)) & ((cmax - cmin) < 0.45 * np.maximum(lum, 1))
    if sel.sum() > 500:
        means = np.array([f[..., i][sel].mean() for i in range(3)])
        gains = np.clip((means.mean() / np.maximum(means, 1)) ** wb, 0.8, 1.3)
        f = f * gains
    lum = luminance(f[..., 0], f[..., 1], f[..., 2])
    # 黑点只拉一部分：最暗的 0.3% 常是噪点、摩尔纹暗线，全拉到纯黑会把深蓝底之类压成黑
    lo = min(float(np.percentile(lum, 0.3)), 40.0) * 0.5
    hi = float(np.percentile(lum, 98.5 if slide else 99.5))
    if hi - lo > 30:
        f = (f - lo) * (250.0 / (hi - lo)) + 3.0
    lum = luminance(f[..., 0], f[..., 1], f[..., 2])
    f = lum[..., None] + (f - lum[..., None]) * vib
    sigma = max(0.7, min(lum.shape) / 2600.0)
    lum2 = unsharp(lum, sigma, sharpen)
    f = f + (lum2 - lum)[..., None]
    out = np.clip(f, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(out[..., ::-1]))


def classify(img):
    """auto 模式：按颜色内容判断用哪种增强。"""
    h, w = img.shape[:2]
    s = min(1.0, 700.0 / max(h, w))
    sm = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = sm[..., 0], sm[..., 1], sm[..., 2]
    lum = luminance(b, g, r)
    bgs = estimate_background([b, g, r], lum)
    nb, ng, nr = (c / bg for c, bg in zip((b, g, r), bgs))
    nl = luminance(nb, ng, nr)
    chroma = np.maximum(np.maximum(nb, ng), nr) - np.minimum(np.minimum(nb, ng), nr)
    colored = ((chroma > 0.18) & (nl > 0.12) & (nl < 0.97)).astype(np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    color_frac = float(colored.mean())
    # 大块深色/彩色区域（照片、色块底纹）
    big = cv2.morphologyEx(((nl < 0.6) | (chroma > 0.18)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    big_frac = float(big.mean())
    # 整张底色本身就不白（彩色证书、深色幻灯片）
    paper_dark = float(np.median(lum)) < 150 or float(np.median(np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r))) > 40
    # 四周有彩色花边（证书、奖状）：至少三条边的外侧窄带里都有成片的彩色
    hh, ww = colored.shape
    bh, bw = max(2, int(0.12 * hh)), max(2, int(0.12 * ww))
    sides = (colored[:bh], colored[-bh:], colored[:, :bw], colored[:, -bw:])
    framed = sum(float(s_.mean()) > 0.015 for s_ in sides) >= 3
    if color_frac > 0.10 or big_frac > 0.08 or paper_dark or framed:
        return "photo"
    if color_frac > 0.003:
        return "color"
    return "doc"


# ─────────────────────────── 单页流程 ───────────────────────────

def parse_corners(s, W, H):
    nums = [float(x) for x in re.split(r"[,;\s]+", s.strip()) if x]
    if len(nums) != 8:
        raise ScanError("--corners 需要 4 个点、8 个数，例如 \"120,80 1900,95 1880,2700 100,2690\"")
    pts = np.array(nums, np.float64).reshape(4, 2)
    if pts.max() <= 1.0:
        pts *= [W, H]
    return order_quad(pts)


def draw_debug(img, q, path):
    H, W = img.shape[:2]
    s = min(1.0, 1600.0 / max(H, W))
    sm = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    if q is not None:
        cv2.polylines(sm, [np.round(q * s).astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), 3, cv2.LINE_AA)
        for i, p in enumerate(q * s):
            cv2.circle(sm, tuple(int(v) for v in p), 9, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(sm, str(i + 1), (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(path, sm, [cv2.IMWRITE_JPEG_QUALITY, 85])


def scan_one(path, args, debug_path=None):
    t0 = time.time()
    img = load_image(path)
    H, W = img.shape[:2]
    info = {"file": os.path.basename(path)}

    q = None
    if args.corners:
        q = parse_corners(args.corners, W, H)
        info["crop"] = "◆ 手动四角"
    elif not args.no_crop:
        q, conf = detect_page(img)
        if q is not None:
            q = refine_quad(img, q)
            info["crop"] = f"◆ 找边 ΔE{conf * 100:.0f}"
        else:
            info["crop"] = "◇ 无边缘·整图"
    else:
        info["crop"] = "◇ 不裁切"
    if debug_path:
        draw_debug(img, q, debug_path)

    mode = args.mode
    ratio, how = None, None
    if q is not None:
        if args.ratio == "auto":
            ratio, how = estimate_ratio(q, W, H)
            how = None
        elif args.ratio != "free":
            ratio, how = float(args.ratio), None
    # 先按原分辨率矫正一次，用来判断模式
    page = warp_page(img, q, ratio, 1.0) if q is not None else img
    if q is None and not args.no_deskew:
        a = estimate_skew(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
        if a:
            page = rotate_small(page, a)
            info["deskew"] = f"{a:+.2f}°"

    if mode == "auto":
        mode = classify(page)
    info["mode"] = mode

    rewarp = False
    if q is not None and args.ratio == "auto" and ratio:
        snapped_ratio, name = snap_ratio(ratio, mode)
        if name:
            ratio, how, rewarp = snapped_ratio, name.replace(" 横向", "↔"), True
    if q is not None:
        info["ratio"] = (how or f"{ratio:.3f}") if ratio else None
        info["ratio_value"] = ratio

    # 文字行拉平 / 书页展平：只对文档类模式做
    field = None
    if mode in ("doc", "color", "bw", "notes") and not args.no_flatten:
        if rewarp:
            page = warp_page(img, q, ratio, 1.0)
        lines, ch = find_text_lines(page)
        if q is not None:
            fw, fh, off, _ = page_geometry(q, ratio, 1.0)
        else:
            fh, fw = page.shape[:2]
            off = 0
        field, st = None, None
        if getattr(args, "book", False):
            f_page = getattr(estimate_ratio, "last_f", None) if q is not None else None
            field, bst = fit_book(lines, fw, fh, off, ch, f_page or 1.2 * max(W, H))
            info["book"] = bst
            if field is None:
                log(f"  ◇ 书页模型没拟合好（{bst}），退回二维拉平")
        if field is None:
            field, st = fit_line_field(lines, fw, fh, off, ch)
            info["flatten"] = st
        info["flattened"] = field is not None

    # 文字类模式适度放大后再增强：笔画边缘更平滑，打印/放大看更清楚
    scale = args.scale
    if scale is None:
        long_side = max(page.shape[:2])
        scale = min(2.0, 2600.0 / long_side) if mode in ("doc", "color", "bw", "notes") and long_side < 2600 else 1.0
    if q is not None and (scale != 1.0 or rewarp or field is not None):
        page = warp_page(img, q, ratio, scale, field)  # 展平 + 透视 + 放大，从原图一次插值到最终尺寸
    else:
        if scale != 1.0:
            page = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if field is not None:
            page = apply_field(page, field)
    if args.rotate:
        page = np.ascontiguousarray(np.rot90(page, k=-(args.rotate // 90) % 4))

    if mode == "doc":
        out = enhance_doc(page, ink=args.ink, scale=scale)
    elif mode == "bw":
        out = enhance_bw(page, ink=args.ink, scale=scale)
    elif mode == "color":
        out = enhance_color(page, ink=args.ink, scale=scale)
    elif mode == "notes":
        out = enhance_notes(page, ink=args.ink, colors=getattr(args, "colors", 8), scale=scale)
        info["palette"] = len(out.getpalette()) // 3 if out.mode == "P" else 1
    elif mode == "photo":
        out = enhance_photo(page)
    elif mode == "slide":
        out = enhance_photo(page, strength=0.6, sharpen=0.55, vib=1.1, slide=True)
    else:
        out = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
    info["size"] = f"{out.width}×{out.height}"
    info["secs"] = time.time() - t0
    return out, info


# ─────────────────────────── 输出 ───────────────────────────

def pdf_text(s):
    return b"<FEFF" + s.encode("utf-16-be").hex().upper().encode() + b">"


def page_layout(w, h, page):
    """返回 (页面宽, 页面高, 图片放置矩形 x, y, dw, dh)，单位 pt。"""
    r = w / h

    def oriented(sz):
        return (sz[0], sz[1]) if h >= w else (sz[1], sz[0])

    if page in ("a4", "letter"):
        pw, ph = oriented(A4_PT if page == "a4" else LETTER_PT)
        margin = 0.0
    elif page == "auto":
        for sz in (A4_PT, LETTER_PT):
            pw, ph = oriented(sz)
            if abs((pw / ph) / r - 1) < 0.03:
                margin = 0.0
                break
        else:
            pw = ph = None
    else:
        pw = ph = None
    if pw is None:  # fit：长边按 A4 长边算物理尺寸
        sc = A4_PT[1] / max(w, h)
        return w * sc, h * sc, 0.0, 0.0, w * sc, h * sc
    sc = min((pw - 2 * margin) / w, (ph - 2 * margin) / h)
    dw, dh = w * sc, h * sc
    return pw, ph, (pw - dw) / 2, (ph - dh) / 2, dw, dh


def stream(data, compress=False, extra=b""):
    if compress:
        data = zlib.compress(data, 9)
        extra += b" /Filter /FlateDecode"
    return b"<< /Length %d " % len(data) + extra + b" >>\nstream\n" + data + b"\nendstream"


# OCR 文字层用的“无字形字体”：字是透明的（渲染模式 3），只负责让 PDF 能搜索、选中、复制。
# 字码 = Unicode（Identity-H，两字节），CJK 宽 1000、其余 500，和下面 text_width 的算法一致。
W_RANGES = (b"[4352 4447 1000 11904 42191 1000 44032 55203 1000 63744 64255 1000 "
            b"65072 65103 1000 65280 65376 1000 65504 65510 1000]")


def is_wide(o):
    return (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF
            or 0xFE30 <= o <= 0xFE4F or 0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6)


def text_width(t):
    return sum(1000 if is_wide(ord(c)) else 500 for c in t) / 1000.0


def glyphless_font():
    """用 fontTools 现做一个只有空字形的 TrueType 字体嵌进 PDF（Acrobat 等阅读器要求字体可用）。没装 fontTools 就不嵌。"""
    try:
        from fontTools.fontBuilder import FontBuilder
        from fontTools.pens.ttGlyphPen import TTGlyphPen
    except Exception:
        return None
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "box"])
    fb.setupCharacterMap({32: "box"})
    pen = TTGlyphPen(None)  # 给字形一个真实外框：字体有了正常的 bbox，CoreGraphics 才肯解析；文字层是透明的，永远不会画出来
    pen.moveTo((0, -200))
    pen.lineTo((0, 800))
    pen.lineTo((500, 800))
    pen.lineTo((500, -200))
    pen.closePath()
    box = pen.glyph()
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "box": box})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "box": (500, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "GlyphLessFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


def tounicode_cmap():
    rngs = [b"<%02X00> <%02XFF> <%02X00>" % (h, h, h) for h in range(256) if not 0xD8 <= h <= 0xDF]
    blocks = [b"%d beginbfrange\n" % len(rngs[i:i + 100]) + b"\n".join(rngs[i:i + 100]) + b"\nendbfrange"
              for i in range(0, len(rngs), 100)]
    return (b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            b"/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            + b"\n".join(blocks)
            + b"\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend")


def text_layer(lines, x0, y0, dw, dh):
    """把 OCR 的每一行按它的四角放到页面上：旋转跟着行走，水平缩放让字正好铺满这一行。"""
    ops = []
    for ln in lines:
        t = "".join(c for c in ln["text"] if 0x20 <= ord(c) <= 0xFFFF and not 0xD800 <= ord(c) <= 0xDFFF)
        if not t.strip():
            continue
        (blx, bly), (brx, bry), _, (tlx, tly) = ln["quad"]
        bl = np.array([x0 + blx * dw, y0 + bly * dh])
        u = np.array([x0 + brx * dw, y0 + bry * dh]) - bl
        v = np.array([x0 + tlx * dw, y0 + tly * dh]) - bl
        L, Hh = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if L < 1 or Hh < 1:
            continue
        ux, uy = u / L
        fs = Hh * 0.85
        tz = 100.0 * L / max(text_width(t) * fs, 1e-3)
        o = bl + (v / Hh) * ((Hh - fs) / 2 + 0.2 * fs)  # 字体框是基线下 0.2em 到上 0.8em，整体居中到这一行
        ops.append(b"BT 3 Tr /OCR %.2f Tf %.2f Tz %.5f %.5f %.5f %.5f %.2f %.2f Tm <%s> Tj ET"
                   % (fs, tz, ux, uy, -uy, ux, o[0], o[1], t.encode("utf-16-be").hex().upper().encode()))
    return b"\n".join(ops)


def write_pdf(path, images, page="auto", quality=90, title="Scan", ocr=None):
    """自己写 PDF：每页一张图，JPEG 原样嵌入（不二次缩放），黑白页无损 Flate；有 OCR 就叠一层透明文字。"""
    objs = {}
    counter = [0]

    def new():
        counter[0] += 1
        return counter[0]

    catalog, pages_id, info_id = new(), new(), new()
    font_id = None
    if ocr and any(ocr):
        font_id, cid_id, desc_id, tu_id, map_id = new(), new(), new(), new(), new()
        ff = glyphless_font()
        ff_id = new() if ff else None
        objs[font_id] = (b"<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H "
                         b"/DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>" % (cid_id, tu_id))
        objs[cid_id] = (b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont "
                        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
                        b"/FontDescriptor %d 0 R /DW 500 /W " % desc_id + W_RANGES
                        + b" /CIDToGIDMap %d 0 R >>" % map_id)
        objs[desc_id] = (b"<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 5 /FontBBox [0 -200 1000 800] "
                         b"/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80"
                         + (b" /FontFile2 %d 0 R" % ff_id if ff else b"") + b" >>")
        objs[tu_id] = stream(tounicode_cmap(), compress=True)
        objs[map_id] = stream(b"\x00\x01" * 65536, compress=True)  # 所有字码都指向 1 号（空）字形
        if ff:
            objs[ff_id] = stream(ff, compress=True, extra=b"/Length1 %d" % len(ff))
    kids = []
    for i, im in enumerate(images):
        img_id, cont_id, page_id = new(), new(), new()
        kids.append(page_id)
        w, h = im.size
        if im.mode == "1":
            data = zlib.compress(im.tobytes(), 9)
            head = b"/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /FlateDecode"
        elif im.mode == "P":  # 索引色：调色板 ≤16 色就按 1/2/4 位打包，再无损压缩
            pal = bytes(im.getpalette()[:768])
            n = max(1, int(np.asarray(im).max()) + 1)
            pal = pal[:3 * n]
            bpc = 1 if n <= 2 else 2 if n <= 4 else 4 if n <= 16 else 8
            data = zlib.compress(pack_bits(np.asarray(im), bpc), 9)
            head = (b"/ColorSpace [/Indexed /DeviceRGB %d <" % (n - 1) + pal.hex().upper().encode()
                    + b">] /BitsPerComponent %d /Filter /FlateDecode" % bpc)
        else:
            buf = io.BytesIO()
            if im.mode == "L":
                im.save(buf, "JPEG", quality=quality, optimize=True)
                cs = b"/DeviceGray"
            else:
                im = im.convert("RGB")
                im.save(buf, "JPEG", quality=quality, optimize=True, subsampling=0)
                cs = b"/DeviceRGB"
            data = buf.getvalue()
            head = b"/ColorSpace " + cs + b" /BitsPerComponent 8 /Filter /DCTDecode"
        objs[img_id] = (b"<< /Type /XObject /Subtype /Image /Width %d /Height %d " % (w, h) + head
                        + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
        pw, ph, x, y, dw, dh = page_layout(w, h, page)
        content = b"q %.3f 0 0 %.3f %.3f %.3f cm /Im0 Do Q" % (dw, dh, x, y)
        if font_id and ocr[i]:
            content += b"\n" + text_layer(ocr[i], x, y, dw, dh)
        objs[cont_id] = stream(content, compress=True)
        res = b"/XObject << /Im0 %d 0 R >>" % img_id
        if font_id:
            res += b" /Font << /OCR %d 0 R >>" % font_id
        objs[page_id] = (b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.3f %.3f] " % (pages_id, pw, ph)
                         + b"/Resources << " + res + b" >> /Contents %d 0 R >>" % cont_id)
    objs[catalog] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id
    objs[pages_id] = (b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids)
                      + b"] /Count %d >>" % len(kids))
    now = datetime.datetime.now().strftime("D:%Y%m%d%H%M%S")
    objs[info_id] = (b"<< /Title " + pdf_text(title) + b" /Producer (Vellum " + VERSION.encode()
                     + b") /Creator (Vellum) /CreationDate (" + now.encode() + b") >>")
    out = io.BytesIO()
    out.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    n = counter[0]
    offsets = [0] * (n + 1)
    for oid in range(1, n + 1):
        offsets[oid] = out.tell()
        out.write(b"%d 0 obj\n" % oid + objs[oid] + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (n + 1))
    for oid in range(1, n + 1):
        out.write(b"%010d 00000 n \n" % offsets[oid])
    out.write(b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n" % (n + 1, catalog, info_id, xref))
    out.write(b"%%EOF\n")
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(out.getvalue())
    os.replace(tmp, path)


def pack_bits(a, bpc):
    """二维 uint8 索引数组按 bpc 位打包成 PDF 要的行对齐字节流。"""
    if bpc == 8:
        return a.astype(np.uint8).tobytes()
    per = 8 // bpc
    h, w = a.shape
    pad = (-w) % per
    if pad:
        a = np.hstack([a, np.zeros((h, pad), a.dtype)])
    a = a.reshape(h, -1, per).astype(np.uint8)
    out = np.zeros(a.shape[:2], np.uint8)
    for j in range(per):
        out |= (a[:, :, j] << (8 - bpc * (j + 1))).astype(np.uint8)
    return out.tobytes()


def save_image(im, path, quality):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        (im.convert("L") if im.mode == "1" else im.convert("RGB") if im.mode == "P" else im).save(path, "JPEG", quality=quality, optimize=True,
                                                           **({"subsampling": 0} if im.mode == "RGB" else {}))
    elif ext in (".tif", ".tiff"):
        im.save(path, "TIFF", compression="group4" if im.mode == "1" else "tiff_lzw")
    else:
        im.save(path, "PNG", optimize=True)


# ─────────────────────────── OCR ───────────────────────────

HERE = os.path.dirname(os.path.realpath(__file__))
OCR_BIN = os.path.join(HERE, "bin", "vellum-ocr")
OCR_SRC = os.path.join(HERE, "tools", "ocr.swift")
DEFAULT_LANGS = "zh-Hans,zh-Hant,en-US"


REVIEW_APP = os.path.join(HERE, "bin", "Vellum.app")
REVIEW_SRC = os.path.join(HERE, "tools", "review.swift")
APP_ID = "local.mactoolkit.vellum"


def swift_tool(src, dst, label):
    """Swift 小工具：没编译（或源码更新了）就用 swiftc 现编一次，约 1 分钟，以后秒开。"""
    if os.path.exists(dst) and (not os.path.exists(src) or os.path.getmtime(dst) >= os.path.getmtime(src)):
        return dst
    if not shutil.which("swiftc") or not os.path.exists(src):
        return None
    log(f"⚙ 首次启用{label}：swiftc 编译中（约 1 分钟，只编这一次）…")
    r = subprocess.run(["swiftc", "-O", "-swift-version", "5", src, "-o", dst], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"⚠ {label}编译失败：{r.stderr.strip()[:300]}")
        return None
    return dst


def ocr_engine():
    return swift_tool(OCR_SRC, OCR_BIN, " OCR 引擎")


def inspect(files, outdir):
    """给检查窗口用：每张图的预览图、找到的四角（0–1 比例）、找边置信度、auto 会选的模式。"""
    os.makedirs(outdir, exist_ok=True)
    res = []
    for i, f in enumerate(files):
        img = load_image(f)
        H, W = img.shape[:2]
        q, conf = detect_page(img)
        if q is not None:
            q = refine_quad(img, q)
        s = min(1.0, 1800.0 / max(H, W))
        prev = os.path.join(outdir, f"src{i}.jpg")
        cv2.imwrite(prev, cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        page = warp_page(img, q, estimate_ratio(q, W, H)[0], 1.0) if q is not None else img
        res.append({"file": os.path.abspath(f), "preview": prev, "width": W, "height": H,
                    "corners": (q / [W, H]).tolist() if q is not None else None,
                    "conf": round(float(conf), 3), "mode": classify(page)})
    print(json.dumps(res, ensure_ascii=False))


def review_app():
    """检查窗口 Vellum.app：一个最小的 .app 包（有名字、有 Dock 图标、能被系统当成正经 App）。"""
    exe = os.path.join(REVIEW_APP, "Contents", "MacOS", "Vellum")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    if swift_tool(REVIEW_SRC, exe, "检查窗口") is None:
        return None
    icns = os.path.join(REVIEW_APP, "Contents", "Resources", "Vellum.icns")
    svg = os.path.join(HERE, "assets", "icon.svg")
    fresh_icon = False
    if os.path.exists(svg) and (not os.path.exists(icns) or os.path.getmtime(icns) < os.path.getmtime(svg)):
        fresh_icon = build_icns(svg, icns)
    plist = os.path.join(REVIEW_APP, "Contents", "Info.plist")
    if fresh_icon or not os.path.exists(plist) or os.path.getmtime(plist) < os.path.getmtime(exe):
        import plistlib
        with open(plist, "wb") as fh:
            plistlib.dump({"CFBundleIdentifier": APP_ID, "CFBundleName": "Vellum", "CFBundleDisplayName": "Vellum",
                           "CFBundleExecutable": "Vellum", "CFBundlePackageType": "APPL",
                           "CFBundleShortVersionString": VERSION, "LSMinimumSystemVersion": "26.0",
                           "NSHighResolutionCapable": True, "CFBundleIconFile": "Vellum"}, fh)
        subprocess.run(["codesign", "-s", "-", "--force", REVIEW_APP], capture_output=True)
    return REVIEW_APP


def build_icns(svg, icns):
    """SVG → 1024 PNG（系统 Quick Look 渲染）→ 各尺寸 iconset → iconutil 打成 .icns。"""
    tmpd = tempfile.mkdtemp(prefix="vellum-icon-")
    try:
        subprocess.run(["qlmanage", "-t", "-s", "1024", "-o", tmpd, svg], capture_output=True)
        png = os.path.join(tmpd, os.path.basename(svg) + ".png")
        if not os.path.exists(png):
            return False
        iconset = os.path.join(tmpd, "Vellum.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            for k in (1, 2):
                px = size * k
                name = f"icon_{size}x{size}{'@2x' if k == 2 else ''}.png"
                subprocess.run(["sips", "-z", str(px), str(px), png, "--out", os.path.join(iconset, name)],
                               capture_output=True)
        os.makedirs(os.path.dirname(icns), exist_ok=True)
        r = subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], capture_output=True)
        return r.returncode == 0
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def launch_review(files):
    app = review_app()
    if app is None:
        raise ScanError("检查窗口不可用（需要 Xcode 命令行工具里的 swiftc，macOS 26 以上）")
    # -W 等窗口关掉再返回（快速操作的转圈和窗口同生命周期），-n 每次开新实例
    r = subprocess.run(["open", "-n", "-W", app, "--args"] + [os.path.abspath(f) for f in files])
    sys.exit(r.returncode)


def reading_order(lines):
    """Vision 的结果排成阅读顺序：按行从上到下，同一行里从左到右。返回 [[行内各段], ...]。"""
    def top(l):
        return max(p[1] for p in l["quad"])

    def bottom(l):
        return min(p[1] for p in l["quad"])

    def left(l):
        return min(p[0] for p in l["quad"])

    rows = []
    for l in sorted(lines, key=lambda l: -top(l)):
        cy = (top(l) + bottom(l)) / 2
        for row in rows:
            if row["bot"] <= cy <= row["top"]:
                row["items"].append(l)
                break
        else:
            rows.append({"top": top(l), "bot": bottom(l), "items": [l]})
    return [sorted(r["items"], key=left) for r in rows]


def run_ocr(images, langs, fast=False):
    eng = ocr_engine()
    if eng is None:
        return None
    tmpd = tempfile.mkdtemp(prefix="vellum-ocr-")
    try:
        paths = []
        for i, im in enumerate(images):
            p = os.path.join(tmpd, f"p{i}.png")
            (im.convert("L") if im.mode == "1" else im.convert("RGB") if im.mode == "P" else im).save(p, compress_level=1)
            paths.append(p)
        r = subprocess.run([eng, "--langs", ",".join(langs)] + (["--fast"] if fast else []) + paths,
                           capture_output=True)
        if r.returncode != 0:
            raise ScanError("OCR 失败：" + r.stderr.decode("utf-8", "replace").strip())
        return [reading_order(p["lines"]) for p in json.loads(r.stdout)]
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def page_text(rows):
    return "\n".join("  ".join(l["text"] for l in row) for row in rows)


def flat(rows):
    return [l for row in rows for l in row]


# ─────────────────────────── 命令行 ───────────────────────────

def log(msg):
    print(msg, file=sys.stderr, flush=True)


def unique_path(p):
    if not os.path.exists(p):
        return p
    stem, ext = os.path.splitext(p)
    i = 2
    while os.path.exists(f"{stem} {i}{ext}"):
        i += 1
    return f"{stem} {i}{ext}"


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def expand_inputs(items):
    files = []
    for it in items:
        if os.path.isdir(it):
            names = sorted((n for n in os.listdir(it) if os.path.splitext(n)[1].lower() in IMAGE_EXTS
                            and not n.startswith(".")), key=natural_key)
            files += [os.path.join(it, n) for n in names]
        elif os.path.isfile(it):
            files.append(it)
        else:
            raise ScanError(f"找不到文件：{it}")
    if not files:
        raise ScanError("没有可处理的图片")
    return files


def notify(title, msg):
    def q(s):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    subprocess.run(["osascript", "-e", f"display notification {q(msg)} with title {q(title)}"], capture_output=True)


def fmt_page(i, n, info):
    parts = [info["crop"]]
    if info.get("ratio"):
        parts.append(info["ratio"])
    parts.append(MODE_NAMES.get(info["mode"], info["mode"]))
    if info.get("flattened") and info.get("book") and "flatten" not in info:
        parts.append(f"书页 3D · 弧长 ×{info['book']['arc']:.3f}")
    elif info.get("flattened"):
        parts.append(f"拉平 {info['flatten']['lines']} 行")
    if info.get("deskew"):
        parts.append(f"纠偏 {info['deskew']}")
    parts += [info["size"], f"{info['secs']:.1f}s"]
    return f"[{i}/{n}] {info['file']}  " + " · ".join(parts)


EPILOG = """模式  -m
  auto    自动识别（默认）
  doc     白底黑字 · 纸面纯白、墨色加深、文字行拉平
  color   彩色文档 · 同上，但保留红章、彩笔、logo 的颜色
  notes   笔记 · 调色板压缩：纸白 + 少量墨色，文件最小还保留彩笔（--colors 调颜色数）
  photo   保真 · 证书、奖状、海报：保留底纹花边，只校光照和偏色
  slide   幻灯片 · 拍屏幕/投影：去摩尔纹，对齐 16:9
  bw      1-bit 黑白 · 体积最小
  none    只裁切矫正，不动颜色

墨色  --ink
  0 raw · 1 normal · 2 deep（默认）· 3 bold

示例
  vellum IMG_0001.HEIC                  → IMG_0001_scan.pdf（带 OCR，可搜索）
  vellum p*.jpg -o 作业.pdf              多张合成一个 PDF
  vellum ~/Desktop/拍照 -o 合同.pdf      整个文件夹，按文件名自然排序
  vellum 笔记.jpg --ink 3                字再黑一点、粗一点
  vellum 合同.jpg --txt                  PDF 之外再导出一份 .txt
  vellum 截图.png --text-only --copy     只要文字：识别完直接进剪贴板
  vellum 投影.jpg -m slide
  vellum IMG.jpg --debug                 另存找边结果图，排障用

全程本机处理：OpenCV 做图像，macOS Vision 做 OCR，零上传。"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="vellum",
        description="Vellum · 手机照片 → 扫描件。找边、透视矫正、文字行拉平、去阴影、墨色增强、OCR，全在本机。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
        add_help=False,
    )
    p._positionals.title = "输入"
    p._optionals.title = "通用"
    p.add_argument("-h", "--help", action="help", help="显示帮助")
    p.add_argument("inputs", nargs="*", help="图片或文件夹（jpg / png / heic / tiff …）")
    g = p.add_argument_group("输出")
    g.add_argument("-o", "--output", help="输出路径：.pdf 合成一份；.png/.jpg/.tif 出图片；.txt 只出文字。"
                                          "默认：第一张图旁边的 <名字>_scan.pdf")
    g.add_argument("--format", choices=("pdf", "png", "jpg", "tif"), help="没给 -o 时的输出格式（默认 pdf）")
    g.add_argument("--each", action="store_true", help="每张图各出一个文件，不合并")
    g.add_argument("--page", choices=("auto", "a4", "letter", "fit"), default="auto",
                   help="PDF 页面：auto（接近 A4/Letter 就用它）· a4 · letter · fit（按图片比例）")
    g.add_argument("--quality", type=int, default=90, help="JPEG 质量，默认 90")
    g.add_argument("--open", action="store_true", help="完成后打开结果")
    e = p.add_argument_group("画质")
    e.add_argument("-m", "--mode", choices=MODES, default="auto", help="处理模式，见下方说明")
    e.add_argument("--colors", type=int, default=8, metavar="N",
                   help="notes 模式的调色板颜色数（含白纸），2–16，默认 8")
    e.add_argument("--ink", type=int, choices=(0, 1, 2, 3), default=2,
                   help="墨色浓度：0 raw · 1 normal · 2 deep（默认）· 3 bold")
    e.add_argument("--scale", type=float, help="输出放大倍数（默认：文字类放大到长边约 2600 像素，最多 2 倍）")
    t = p.add_argument_group("几何")
    t.add_argument("--ratio", type=parse_ratio, default="auto",
                   help="页面宽高比：auto（估计并吸附 A4/Letter/16:9）· free · A4 · 16:9 · 宽:高")
    t.add_argument("--no-crop", action="store_true", help="不找页面边缘（输入已经是扫描件/截图）")
    t.add_argument("--corners", help="手动四角（像素或 0–1 比例），自动找边失败时用")
    t.add_argument("--no-flatten", action="store_true", help="不拉平文字行、不展平书页")
    t.add_argument("--book", action="store_true",
                   help="书页模式：3D 三次曲面 + 非线性优化（SciPy），把书脊附近被压窄的字拉回原宽；拟合不好自动退回二维拉平")
    t.add_argument("--no-deskew", action="store_true", help="不裁切时也不纠正小角度倾斜")
    t.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="输出顺时针旋转")
    o = p.add_argument_group("OCR（macOS Vision，离线）")
    o.add_argument("--ocr", dest="ocr", action="store_true", default=None, help="强制开启 OCR（输出 PDF 时默认就开）")
    o.add_argument("--no-ocr", dest="ocr", action="store_false", help="关闭 OCR")
    o.add_argument("--lang", default=DEFAULT_LANGS, help=f"识别语言，按优先级逗号分隔，默认 {DEFAULT_LANGS}")
    o.add_argument("--fast-ocr", action="store_true", help="快速识别（更快，准确率低一些）")
    o.add_argument("--txt", action="store_true", help="同时导出 .txt")
    o.add_argument("--copy", action="store_true", help="识别出的文字复制到剪贴板")
    o.add_argument("--text-only", action="store_true", help="只要文字：不生成扫描文件，文字打印到标准输出")
    m = p.add_argument_group("其它")
    m.add_argument("--notify", action="store_true", help="完成/出错时发系统通知（快速操作用）")
    m.add_argument("--debug", action="store_true", help="另存一张标出页面四角的预览图")
    r = p.add_argument_group("检查窗口")
    r.add_argument("--review", action="store_true", help="打开检查窗口：拖四角、切模式和墨色、实时预览，回车导出")
    r.add_argument("--review-if-unsure", action="store_true", help="有任何一页没找到边缘就打开检查窗口，否则照常导出")
    r.add_argument("--inspect", metavar="DIR", help=argparse.SUPPRESS)  # 窗口内部用：输出检测结果 JSON
    r.add_argument("--plan", metavar="JSON", help=argparse.SUPPRESS)    # 窗口内部用：逐页四角/模式/墨色
    m.add_argument("--version", action="version", version=f"Vellum {VERSION}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_all = time.time()
    try:
        plan = None
        if args.plan:
            with open(args.plan, encoding="utf-8") as fh:
                plan = json.load(fh)["pages"]
            files = [pg["file"] for pg in plan]
        else:
            files = expand_inputs(args.inputs)
        if args.inspect:
            inspect(files, args.inspect)
            return 0
        if args.review:
            launch_review(files)
        if args.review_if_unsure and any(detect_page(load_image(f))[0] is None for f in files):
            log("◇ 有页面没找到边缘 → 打开检查窗口")
            if args.notify:
                notify("Vellum", "有页面没找到边缘，请在窗口里拖一下四角")
            launch_review(files)
        fmt = args.format
        if args.output:
            ext = os.path.splitext(args.output)[1].lower().lstrip(".")
            fmt = {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext)
            if fmt == "txt":
                args.text_only, args.txt = True, True
            elif fmt not in ("pdf", "png", "jpg", "tif"):
                raise ScanError("输出扩展名要是 .pdf / .png / .jpg / .tif / .txt")
        fmt = fmt or "pdf"
        per_file = args.each or fmt not in ("pdf", "txt")
        langs = [x.strip() for x in args.lang.split(",") if x.strip()]
        want_ocr = bool(args.text_only or args.txt or args.copy or args.ocr
                        or (args.ocr is None and fmt == "pdf"))

        log(f"⚡ Vellum {VERSION} · {len(files)} 页 · mode {args.mode} · ink {INK_NAMES[args.ink]}"
            f" · ocr {'+'.join(dict.fromkeys(l.split('-')[0] for l in langs)) if want_ocr else 'off'}")

        def default_out(src, ext):
            stem = os.path.splitext(os.path.basename(src))[0]
            return os.path.join(os.path.dirname(os.path.abspath(src)), f"{stem}_scan.{ext}")

        out_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else None
        scans = []
        for i, f in enumerate(files, 1):
            dbg = None
            if args.debug:
                stem = os.path.splitext(os.path.basename(f))[0]
                dbg = unique_path(os.path.join(out_dir or os.path.dirname(os.path.abspath(f)), f"{stem}_debug.jpg"))
            a = args
            if plan:  # 检查窗口给的逐页设置
                pg = plan[i - 1]
                a = copy.copy(args)
                a.mode = pg.get("mode", args.mode)
                a.ink = int(pg.get("ink", args.ink))
                a.rotate = int(pg.get("rotate", 0)) % 360
                a.book = bool(pg.get("book", args.book))
                if pg.get("corners"):
                    a.corners = " ".join(f"{x:.6f},{y:.6f}" for x, y in pg["corners"])
            im, info = scan_one(f, a, dbg)
            log(fmt_page(i, len(files), info))
            scans.append((f, im))

        ocr = None
        if want_ocr:
            t0 = time.time()
            ocr = run_ocr([im for _, im in scans], langs, fast=args.fast_ocr)
            if ocr is None:
                log("⚠ 没有 OCR 引擎（需要 Xcode 命令行工具里的 swiftc），这次不做文字识别")
            else:
                nchar = sum(len(l["text"]) for rows in ocr for l in flat(rows))
                log(f"[ocr] {len(scans)} 页 · {nchar} 字 · {time.time() - t0:.1f}s")
        texts = [page_text(rows) for rows in ocr] if ocr else None

        # 组织输出：(路径, [页索引])
        groups = []
        if not args.text_only:
            if per_file:
                for i, (f, _) in enumerate(scans):
                    if args.output and len(scans) == 1:
                        out = args.output
                    elif args.output:
                        stem, ext = os.path.splitext(args.output)
                        out = f"{stem}-{i + 1}{ext}"
                    else:
                        out = unique_path(default_out(f, fmt))
                    groups.append((out, [i]))
            else:
                groups.append((args.output or unique_path(default_out(scans[0][0], "pdf")), list(range(len(scans)))))

        saved = []
        for out, idx in groups:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            if out.lower().endswith(".pdf"):
                write_pdf(out, [scans[i][1] for i in idx], page=args.page, quality=args.quality,
                          title=os.path.splitext(os.path.basename(out))[0],
                          ocr=[flat(ocr[i]) for i in idx] if ocr else None)
            else:
                save_image(scans[idx[0]][1], out, args.quality)
            saved.append(out)
            tag = " · 可搜索" if ocr and out.lower().endswith(".pdf") else ""
            log(f"✓ {out}  {len(idx)} 页 · {os.path.getsize(out) / 1024:.0f} KB{tag}")

        full_text = "\n\n".join(texts) if texts else ""
        if texts and args.txt:
            if args.output and args.output.lower().endswith(".txt"):
                tpaths = [(args.output, list(range(len(scans))))]
            elif groups:
                tpaths = [(os.path.splitext(o)[0] + ".txt", idx) for o, idx in groups]
            else:
                tpaths = [(unique_path(os.path.splitext(default_out(scans[0][0], "txt"))[0] + ".txt"),
                           list(range(len(scans))))]
            for tp, idx in tpaths:
                with open(tp, "w", encoding="utf-8") as fh:
                    fh.write("\n\n".join(texts[i] for i in idx) + "\n")
                saved.append(tp)
                log(f"✓ {tp}  文字 {sum(len(texts[i]) for i in idx)} 字")
        if args.text_only and not (args.output and args.output.lower().endswith(".txt")):
            print(full_text)
        if texts and args.copy:
            subprocess.run(["pbcopy"], input=full_text.encode("utf-8"))
            log(f"✓ 已复制到剪贴板 · {len(full_text)} 字")

        log(f"⏱ {time.time() - t_all:.1f}s")
        if args.notify:
            if args.text_only:
                msg = f"已复制 {len(full_text)} 字到剪贴板" if args.copy else f"识别出 {len(full_text)} 字"
                notify("Vellum · 提取文字", ("✓ " + msg) if texts else "没有识别出文字")
            else:
                tag = " · 可搜索" if ocr else ""
                notify("Vellum · 扫描完成", f"✓ {os.path.basename(saved[0])} · {len(scans)} 页{tag}")
        if args.open and saved:
            pdfs = [s for s in saved if not s.endswith(".txt")]
            target = pdfs or saved
            subprocess.run(["open", target[0]] if len(target) == 1 else ["open", "-R", target[0]])
        return 0
    except ScanError as e:
        log(f"✗ vellum：{e}")
        if args.notify:
            notify("Vellum · 失败", str(e))
        return 2
    except Exception as e:  # 给“快速操作”一个看得见的错误
        if args.notify:
            notify("Vellum · 失败", f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
