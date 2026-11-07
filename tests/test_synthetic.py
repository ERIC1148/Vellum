#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成测试：画出已知内容的文件/证书/幻灯片，按真实相机透视“拍”到桌面上，
加阴影、偏色、噪声和 JPEG 压缩，再跑 vellum，检查找边精度、长宽比和模式判断。

用法：
  python3 tests/test_synthetic.py            # 跑测试，结果和对比图放到临时目录
  python3 tests/test_synthetic.py -o 目录     # 指定输出目录
"""
import argparse
import math
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vellum as ds  # noqa: E402

FONT_LATIN = "/System/Library/Fonts/Helvetica.ttc"
FONT_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_SERIF = "/System/Library/Fonts/Times.ttc"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


# ───────────── 原稿 ─────────────

def doc_text(rng):
    W, H = 1654, 2339  # A4 @200dpi
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((150, 140), "Quarterly Engineering Report", fill="black", font=font(FONT_LATIN, 64))
    d.text((150, 240), "工程季度报告 · 第三季度", fill="black", font=font(FONT_CJK, 48))
    f = font(FONT_SERIF, 34)
    y = 360
    words = ("thermal load bearing motor torque efficiency stress strain vibration frequency "
             "battery module coolant flow design review margin").split()
    for i in range(30):
        line = " ".join(rng.choice(words) for _ in range(9))
        d.text((150, y), line, fill=(20, 20, 20), font=f)
        y += 50
        if i == 14:  # 表格
            for r in range(5):
                d.line((150, y + r * 50, 1500, y + r * 50), fill="black", width=2)
            for c in (150, 600, 1050, 1500):
                d.line((c, y, c, y + 200), fill="black", width=2)
            y += 240
    d.rectangle((1250, 2050, 1500, 2200), fill=(15, 15, 15))  # 黑色 logo 块
    return im


def doc_color(rng):
    im = doc_text(rng)
    d = ImageDraw.Draw(im)
    # 红色圆章
    cx, cy, r = 1200, 1900, 170
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 20, 30), width=12)
    d.text((cx - 120, cy - 30), "已审核 APPROVED", fill=(200, 20, 30), font=font(FONT_CJK, 40))
    # 蓝色批注
    pts = [(200 + i * 14, 1700 + 25 * math.sin(i / 3)) for i in range(60)]
    d.line(pts, fill=(20, 60, 200), width=6)
    d.text((200, 1760), "请在周五前确认", fill=(20, 60, 200), font=font(FONT_CJK, 44))
    return im


def certificate(rng):
    W, H = 2339, 1654  # A4 横向
    im = Image.new("RGB", (W, H), (246, 238, 214))
    d = ImageDraw.Draw(im)
    for i, col in enumerate(((150, 20, 30), (200, 160, 60), (150, 20, 30))):
        m = 50 + i * 28
        d.rectangle((m, m, W - m, H - m), outline=col, width=14 if i != 1 else 6)
    d.text((W / 2 - 330, 230), "荣 誉 证 书", fill=(170, 20, 30), font=font(FONT_CJK, 150))
    d.text((W / 2 - 520, 520), "授予 张同学 在 2026 年度工程设计竞赛中", fill=(40, 30, 20), font=font(FONT_CJK, 64))
    d.text((W / 2 - 300, 640), "荣获 一等奖 特发此证", fill=(40, 30, 20), font=font(FONT_CJK, 64))
    d.ellipse((1650, 1050, 2000, 1400), fill=(210, 40, 40))
    d.text((1700, 1190), "评审委员会", fill=(250, 220, 200), font=font(FONT_CJK, 52))
    d.text((300, 1300), "Certificate of Excellence", fill=(120, 90, 40), font=font(FONT_SERIF, 72))
    return im


def slide(rng):
    W, H = 1920, 1080
    im = Image.new("RGB", (W, H), (18, 40, 92))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 170), fill=(230, 90, 30))
    d.text((80, 40), "Battery Thermal Strategy", fill="white", font=font(FONT_LATIN, 90))
    for i, t in enumerate(("Side cooling: h = 216 W/m²K", "Bottom cooling: h = 301 W/m²K", "Cost: $6.0 vs $6.5")):
        d.ellipse((100, 300 + i * 140, 140, 340 + i * 140), fill=(250, 200, 60))
        d.text((180, 285 + i * 140), t, fill="white", font=font(FONT_LATIN, 64))
    for i, (hgt, col) in enumerate(((300, (90, 200, 120)), (420, (250, 200, 60)), (220, (230, 90, 30)))):
        d.rectangle((1300 + i * 170, 900 - hgt, 1420 + i * 170, 900), fill=col)
    return im


# ───────────── “拍照” ─────────────

def background(kind, W, H, rng):
    if kind == "wood":
        x = np.linspace(0, 1, W)[None, :]
        y = np.linspace(0, 1, H)[:, None]
        grain = np.sin((x * 40 + np.sin(y * 7) * 2) * math.pi) * 0.5 + 0.5
        base = np.dstack([45 + 25 * grain, 75 + 30 * grain, 110 + 35 * grain])  # BGR 棕色
    elif kind == "dark":
        base = np.full((H, W, 3), (40, 38, 36), np.float64)
    else:  # 浅灰桌面：最难
        base = np.full((H, W, 3), (170, 172, 175), np.float64)
    noise = rng.normal(0, 6, (H // 8, W // 8, 1))
    base += cv2.resize(noise, (W, H))[..., None] if noise.ndim == 2 else cv2.resize(noise, (W, H))[..., None]
    return base


def photograph(src, rng, bg="wood", tilt=(18, -10, 6), shadow=True, cast=(0.86, 0.95, 1.0), screen=False):
    """把原稿放到 3D 空间里，用针孔相机（焦距≈1.1×长边，主点在中心）投影。返回 照片, 真实四角。"""
    W, H = 3000, 4000
    if src.width > src.height:
        W, H = 4000, 3000
    f = 1.1 * max(W, H)
    ax, ay, az = (math.radians(a) for a in tilt)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    w, h = src.size
    s = 1.0 / max(w, h)
    obj = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]]) * s
    Z = 1.75
    cam = (R @ obj.T).T + [rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), Z]
    pts = np.stack([f * cam[:, 0] / cam[:, 2] + W / 2, f * cam[:, 1] / cam[:, 2] + H / 2], 1)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), pts.astype(np.float32))
    srcbgr = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
    if screen:  # 屏幕像素网格 → 摩尔纹
        grid = np.ones((h, w), np.float32)
        grid[::3, :] = 0.8
        grid[:, ::3] *= 0.85
        srcbgr = (srcbgr * grid[..., None]).astype(np.uint8)
    warped = cv2.warpPerspective(srcbgr, M, (W, H), flags=cv2.INTER_AREA if not screen else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = background(bg, W, H, rng)
    if screen:  # 屏幕边框
        bez = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
        bez = cv2.dilate(bez, np.ones((90, 90), np.uint8))
        out[bez > 0] = (15, 15, 15)
    m = (mask.astype(np.float64) / 255)[..., None]
    out = out * (1 - m) + warped.astype(np.float64) * m
    # 光照：一侧亮一侧暗 + 暗角
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    light = 0.78 + 0.22 * (xx / W) + 0.08 * (yy / H)
    light *= 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    if shadow:  # 手机/手的软阴影
        sh = np.zeros((H, W), np.uint8)
        cv2.fillPoly(sh, [np.int32([[0, H * 0.55], [W * 0.45, H * 0.62], [W * 0.5, H], [0, H]])], 255)
        sh = cv2.GaussianBlur(sh, (0, 0), 60).astype(np.float64) / 255
        light *= 1 - 0.38 * sh
    out *= light[..., None]
    if not screen:
        out *= np.array(cast)[None, None, :]
    out += rng.normal(0, 3.0, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, pts


def curved(make, amp=0.035, slant=0.012):
    """书页/卷起的纸：内容按 x² 弯曲（下半页弯得更厉害）再加一点整体倾斜。纸的外轮廓保持矩形。"""
    def f(rng):
        im = np.asarray(make(rng))
        h, w = im.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xn, yn = (X - w / 2) / (w / 2), (Y - h / 2) / (h / 2)
        d = (amp * h * xn ** 2 * (0.7 + 0.3 * yn) + slant * h * xn).astype(np.float32)
        out = cv2.remap(im, X, Y - d, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(out)
    return f


def ruled_page(rng):
    """书页测试稿：顶部一排等距刻度（量“书脊附近被压窄”），下面是文字。"""
    im = doc_text(rng)
    d = ImageDraw.Draw(im)
    for x in range(150, 1510, 60):
        d.rectangle((x - 3, 60, x + 3, 120), fill="black")
    return im


def book_curl(make, a=1.1, b=-0.25, f_ratio=1.75 * math.sqrt(2)):
    # f_ratio = 相机到纸面距离 / 页宽，和 photograph() 的相机一致（Z=1.75，纸长边归一化为 1，A4 宽 = 1/√2）
    """把原稿贴到 z(u)=(a+b)u³−(2a+b)u²+au 的曲面上，用针孔相机正对着看（摊开的书页，书脊在左）。"""
    def f(rng):
        src = np.asarray(make(rng))
        H, W = src.shape[:2]
        D = f_ratio
        Xg = np.linspace(-0.5, 0.5, 6001)
        Zg = ds.book_z(Xg + 0.5, a, b)
        xg = D * Xg / (D - Zg)
        span = xg.max() - xg.min()
        ow = int(span * W) + 40
        oh = int(H * D / (D - Zg.max())) + 40
        xo = (np.arange(ow) - ow / 2) / W + (xg.max() + xg.min()) / 2
        X = np.interp(xo, xg, Xg)
        Z = ds.book_z(X + 0.5, a, b)
        # 纸不能拉伸：原稿按“沿曲面的弧长”贴上去，书脊附近看起来就被压窄了
        sg = np.concatenate([[0], np.cumsum(np.hypot(np.diff(Xg), np.diff(Zg)))])
        S_ = np.interp(X, Xg, sg) / sg[-1]
        yo = (np.arange(oh) - oh / 2) / W
        mx = np.broadcast_to((S_ * W)[None, :], (oh, ow)).astype(np.float32)
        my = (yo[:, None] * (D - Z[None, :]) / D * W + H / 2).astype(np.float32)
        out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(out)
    return f


def tick_spacing(img_pil):
    """输出图顶部刻度的 x 位置（只留又细又高的竖线，排除文字）。"""
    g = np.asarray(img_pil.convert("L"))
    h, w = g.shape
    top = (g[: int(0.15 * h)] < 128).astype(np.uint8)
    k = max(5, int(0.022 * h))
    vert = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(vert)
    xs = sorted(cen[i][0] for i in range(1, n) if st[i, 2] < 0.012 * w and st[i, 3] >= k)
    return np.array(xs)


def tick_uniformity(img_pil):
    """刻度间距的变异系数（越小越均匀；被书脊压窄会变大）和左右两端间距之比。"""
    xs = tick_spacing(img_pil)
    if len(xs) < 6:
        return float("nan"), len(xs)
    d = np.diff(xs)
    return float(np.std(d) / np.mean(d)), len(xs)


def straightness(img_pil):
    """输出图里文字行的平均倾斜（°）、行斜率均方根（°）、行内弯曲（相对字高）。"""
    g = cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    lines, ch = ds.find_text_lines(g)
    sl, bend, wts = [], [], []
    for xs, ys in lines:
        if np.ptp(xs) < 8 * ch or xs.size < 8:
            continue
        c2 = np.polyfit(xs, ys, 2)
        c1 = np.polyfit(xs, ys, 1)
        sl.append(math.degrees(math.atan(c1[0])))
        # 二次项在整段上造成的弓高
        half = np.ptp(xs) / 2
        bend.append(abs(c2[0]) * half ** 2 / ch)
        wts.append(np.ptp(xs))
    if not sl:
        return 0.0, 0.0, 0.0, 0
    sl, bend, wts = np.array(sl), np.array(bend), np.array(wts)
    return (float(np.average(sl, weights=wts)), float(np.sqrt(np.average(sl ** 2, weights=wts))),
            float(np.average(bend, weights=wts)), len(sl))


CASES = [
    # 名字, 原稿, 拍摄参数, 期望模式, 期望比例（宽/高）
    ("文档_木桌_阴影", doc_text, dict(bg="wood", tilt=(20, -8, 5)), "doc", 1 / math.sqrt(2)),
    ("文档_浅灰桌面", doc_text, dict(bg="light", tilt=(-12, 10, -4), shadow=False), "doc", 1 / math.sqrt(2)),
    ("书页_弯曲倾斜", curved(doc_text), dict(bg="wood", tilt=(14, -6, 3)), "doc", 1 / math.sqrt(2)),
    ("彩色文档_印章", doc_color, dict(bg="dark", tilt=(15, 12, 8)), "color", 1 / math.sqrt(2)),
    ("证书_横向", certificate, dict(bg="wood", tilt=(-16, -12, -3)), "photo", math.sqrt(2)),
    ("幻灯片_拍屏幕", slide, dict(bg="dark", tilt=(8, 22, 2), shadow=False, screen=True), "photo", 16 / 9),
]


BOOK_CASE = ("书页_摊开书_3D", book_curl(ruled_page), dict(bg="wood", tilt=(10, -5, 2), shadow=False))


class Args:  # 和命令行默认值一致
    mode = "auto"
    ratio = "auto"
    corners = None
    no_crop = False
    no_deskew = False
    no_flatten = False
    scale = None
    rotate = 0
    ink = 2
    colors = 8
    book = False


def corner_error(found, truth):
    found = ds.order_quad(found)
    truth = ds.order_quad(truth)
    diag = np.linalg.norm(truth[2] - truth[0])
    return float(np.max(np.linalg.norm(found - truth, axis=1))), float(np.max(np.linalg.norm(found - truth, axis=1)) / diag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="vellum-test-")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    ok_all = True
    thumbs = []
    print(f"输出目录：{out}")
    for name, make, kw, want_mode, want_ratio in CASES:
        src = make(rng)
        photo, truth = photograph(src, rng, **kw)
        p_in = os.path.join(out, f"{name}_拍照.jpg")
        cv2.imwrite(p_in, photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img = ds.load_image(p_in)
        q, conf = ds.detect_page(img)
        if q is None:
            print(f"✗ {name}: 没找到页面")
            ok_all = False
            continue
        q = ds.refine_quad(img, q)
        err_px, err_rel = corner_error(q, truth)
        r, how = ds.estimate_ratio(q, img.shape[1], img.shape[0])
        im, info = ds.scan_one(p_in, Args())
        mode = info["mode"]
        snapped = info.get("ratio_value") or r
        ratio_err = abs(snapped / want_ratio - 1)
        ok = err_rel < 0.004 and ratio_err < 0.02 and mode == want_mode
        extra = ""
        if want_mode in ("doc", "color"):  # 文档：输出里每行都要水平、不弯
            mean_sl, rms_sl, bend, nl = straightness(im)
            line_ok = abs(mean_sl) < 0.3 and rms_sl < 0.5 and bend < 0.15
            ok &= line_ok
            extra = f" · 行 {nl} 条：平均倾斜 {mean_sl:+.2f}° 斜率均方根 {rms_sl:.2f}° 弓高 {bend:.2f} 字高"
            if "flatten" in info and "before" in info["flatten"]:
                f = info["flatten"]
                extra += f"（展平 {'是' if info['flattened'] else '否'}，残差 {f['before']:.1f}→{f['after']:.1f}px）"
        ok_all &= ok
        print(f"{'✓' if ok else '✗'} {name}: 角点最大误差 {err_px:.1f}px（对角线的 {err_rel * 100:.2f}%）"
              f" · 比例 估计 {r:.3f}/{how} → 输出 {snapped:.3f}，真值 {want_ratio:.3f}（误差 {ratio_err * 100:.1f}%）"
              f" · 模式 {mode}（期望 {want_mode}）{extra} · {info['secs']:.1f}s")
        p_out = os.path.join(out, f"{name}_扫描.jpg")
        ds.save_image(im, p_out, 90)
        t_in = Image.open(p_in)
        t_in.thumbnail((700, 700))
        t_out = im.convert("RGB")
        t_out.thumbnail((700, 700))
        thumbs.append((t_in, t_out))
    # 书页 3D 模型：同一张“摊开的书”分别用二维拉平和 --book，比较刻度间距均匀度和行直度
    name, make, kw = BOOK_CASE
    photo, _ = photograph(make(rng), rng, **kw)
    p_in = os.path.join(out, f"{name}_拍照.jpg")
    cv2.imwrite(p_in, photo, [cv2.IMWRITE_JPEG_QUALITY, 88])

    class BookArgs(Args):
        book = True

    im2, info2 = ds.scan_one(p_in, Args())
    im3, info3 = ds.scan_one(p_in, BookArgs())
    cv2_, n2 = tick_uniformity(im2)
    cv3, n3 = tick_uniformity(im3)
    s2, s3 = straightness(im2), straightness(im3)
    used3d = "book" in info3 and info3.get("flattened") and "flatten" not in info3
    ok = used3d and cv3 < 0.6 * cv2_ and s3[2] < 0.2
    ok_all &= ok
    print(f"{'✓' if ok else '✗'} {name}: 刻度间距变异 二维 {cv2_:.3f}（{n2} 个）→ 3D {cv3:.3f}（{n3} 个）"
          f" · 弓高 {s2[2]:.2f} → {s3[2]:.2f} 字高 · 3D 拟合 {info3.get('book')} · {info3['secs']:.1f}s")
    ds.save_image(im2, os.path.join(out, f"{name}_二维.jpg"), 90)
    ds.save_image(im3, os.path.join(out, f"{name}_3D.jpg"), 90)

    # 对比图
    if thumbs:
        Wt = 1420
        Ht = sum(max(a.height, b.height) + 10 for a, b in thumbs)
        sheet = Image.new("RGB", (Wt, Ht), (128, 128, 128))
        y = 0
        for a, b in thumbs:
            sheet.paste(a, (0, y))
            sheet.paste(b, (720, y))
            y += max(a.height, b.height) + 10
        sheet.save(os.path.join(out, "对比.jpg"), quality=85)
    print("全部通过" if ok_all else "有失败项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
