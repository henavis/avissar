import io
import csv
import cv2
import math
import time
import random
import datetime
import numpy as np
from PIL import Image
import streamlit as st

# ========= קבועים פיזיים =========
WIDTH_MM, HEIGHT_MM = 700, 500   # משטח עבודה במ"מ (שומר יחס וממקם באמצע)

# ========= פונקציות עזר =========
def np_bgr_to_pil_rgb(arr_bgr):
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))

def np_gray_to_pil(gray):
    return Image.fromarray(gray)

def resize_to_physical_dimensions_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
    """מקבל תמונה (BGR או GRAY), משנה גודל לפי dpi ושומר יחס, ממקם על קנבס שחור במרכז"""
    target_w_px, target_h_px = int(width_mm * dpi), int(height_mm * dpi)
    ih, iw = image.shape[:2]
    aspect_img = iw / ih
    aspect_target = target_w_px / target_h_px

    if aspect_img > aspect_target:
        new_w = target_w_px
        new_h = max(1, int(target_w_px / aspect_img))
    else:
        new_h = target_h_px
        new_w = max(1, int(target_h_px * aspect_img))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 3:
        canvas = np.zeros((target_h_px, target_w_px, 3), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h_px, target_w_px), dtype=image.dtype)

    y_off = (target_h_px - new_h) // 2
    x_off = (target_w_px - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ===== עיבודי תמונה =====
def adjust_brightness_contrast(img_gray, brightness=0, contrast=1.0):
    res = img_gray.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(res, 0, 255).astype(np.uint8)

def adjust_gamma(img_gray, gamma=1.0):
    gamma = max(0.01, float(gamma))
    inv = 1.0 / gamma
    norm = img_gray.astype(np.float32) / 255.0
    out = np.power(norm, inv) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def gaussian_blur(img_gray, sigma=0.0):
    if sigma <= 0: 
        return img_gray
    k = int(max(3, 2*round(3*sigma)+1))
    return cv2.GaussianBlur(img_gray, (k, k), sigma)

def unsharp_mask(img_gray, amount=0.0, sigma=1.0):
    if amount <= 0:
        return img_gray
    blur = gaussian_blur(img_gray, sigma)
    sharp = img_gray.astype(np.float32) + amount*(img_gray.astype(np.float32) - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_CLAHE(img_gray, clip=0.0, tile=8):
    if clip <= 0:
        return img_gray
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(img_gray)

# ===== Stipple (שכבה אחת, בהיר=נקודות גדולות יותר) =====
def stipple_layer(gray, dpi, cell_size_mm, max_dots, sensitivity,
                  min_dia_mm, max_dia_mm, seed=None, flip_y=False):
    if seed is not None:
        random.seed(int(seed))

    h, w = gray.shape
    cell_px = max(1, int(cell_size_mm * dpi))

    stipple_img = np.zeros((h, w), dtype=np.uint8)
    points_mm = []

    for y in range(0, h, cell_px):
        for x in range(0, w, cell_px):
            cell = gray[y:y+cell_px, x:x+cell_px]
            if cell.size == 0:
                continue
            mean_val = float(np.mean(cell))
            frac = mean_val / 255.0  # 0=שחור, 1=לבן
            num_dots = int(max_dots * (frac ** sensitivity))  # בהיר -> יותר נקודות אם sensitivity גבוה

            ch, cw = cell.shape[:2]
            for _ in range(num_dots):
                rx = random.randint(0, max(0, cw - 1))
                ry = random.randint(0, max(0, ch - 1))
                cx, cy = x + rx, y + ry
                if 0 <= cx < w and 0 <= cy < h:
                    intensity = gray[cy, cx] / 255.0  # 0=שחור, 1=לבן
                    # בהיר = נקודות גדולות יותר
                    dia_mm = min_dia_mm + intensity * (max_dia_mm - min_dia_mm)
                    radius_px = max(1, int((dpi * dia_mm) / 2))
                    cv2.circle(stipple_img, (cx, cy), radius_px, (255,), -1)

                    px_mm, py_mm = cx / dpi, cy / dpi
                    if flip_y:
                        py_mm = HEIGHT_MM - py_mm
                    points_mm.append((px_mm, py_mm, dia_mm))

    return stipple_img, points_mm

# ===== Path Optimization (Nearest Neighbor) =====
def optimize_path(points_xy):
    if not points_xy:
        return []
    pts = points_xy.copy()
    path = [pts.pop(0)]
    while pts:
        last = path[-1]
        nearest = min(pts, key=lambda p: (p[0]-last[0])**2 + (p[1]-last[1])**2)
        path.append(nearest)
        pts.remove(nearest)
    return path

# ===== ייצוא =====
def build_svg(points_mm, width_mm=WIDTH_MM, height_mm=HEIGHT_MM):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{width_mm}x{height_mm}_{ts}.svg"
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">\n'
    bg = f'  <rect x="0" y="0" width="{width_mm}" height="{height_mm}" fill="black"/>\n'
    circles = [f'  <circle cx="{x:.3f}" cy="{y:.3f}" r="{dia/2:.3f}" fill="white"/>\n' for (x,y,dia) in points_mm]
    footer = '</svg>\n'
    svg_str = header + bg + "".join(circles) + footer
    return fname, svg_str.encode("utf-8")

def build_csv(points_mm):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.csv"
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["X_mm","Y_mm","Dia_mm"])
    for (x,y,dia) in points_mm:
        writer.writerow([f"{x:.3f}", f"{y:.3f}", f"{dia:.3f}"])
    return fname, output.getvalue().encode("utf-8")

def build_gcode(points_mm, servo_up=30, servo_down=90, dwell_up=80, dwell_down=120, feedrate=3000):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.gcode"

    optimized = optimize_path([(x,y) for (x,y,_) in points_mm])

    lines = []
    lines.append("(Stipple export for GRBL with Path Optimization)")
    lines.append("G21 (mm)")
    lines.append("G90 (abs)")
    lines.append(f"F{int(feedrate)}")
    lines.append(f"M3 S{int(servo_up)}")
    lines.append(f"G4 P{int(dwell_up)}")

    for (x,y) in optimized:
        lines.append(f"G0 X{x:.3f} Y{y:.3f}")
        lines.append(f"M3 S{int(servo_down)}")
        lines.append(f"G4 P{int(dwell_down)}")
        lines.append(f"M3 S{int(servo_up)}")
        lines.append(f"G4 P{int(dwell_up)}")

    lines.append("G0 X0 Y0")
    lines.append("M5")
    lines.append("(End)")

    return fname, "\n".join(lines).encode("utf-8")

# ========= UI =========
st.set_page_config(page_title="Stipple Art Generator", layout="wide")
st.title("🎯 Stipple Art Generator (SVG / CSV / GCODE) — Streamlit")

with st.sidebar:
    st.header("1) העלאת תמונה")
    up = st.file_uploader("בחר/י קובץ תמונה", type=["png","jpg","jpeg","bmp","tif","tiff"])
    st.caption("המערכת תעבוד בגווני אפור בלבד (ממירה אוטומטית).")

    st.header("2) פרמטרים פיזיים")
    dpi = st.slider("DPI (px/mm)", 2, 20, 5, 1)
    flip_y = st.checkbox("Flip Y (ציר הפוך)", value=False)
    fix_seed = st.checkbox("Fix Seed (תוצאה דטרמיניסטית)", value=True)

    st.header("3) פרמטרי Stipple")
    cell_size_mm = st.slider("Cell Size (mm)", 1, 20, 3, 1)
    max_dots = st.slider("Max Dots / Cell", 1, 200, 60, 1)
    sensitivity = st.slider("Sensitivity", 0.2, 4.0, 2.0, 0.1)
    min_dia_mm = st.slider("Min Dot Diameter (mm)", 0.05, 2.0, 0.2, 0.05)
    max_dia_mm = st.slider("Max Dot Diameter (mm)", 0.1, 4.0, 1.5, 0.1)
    if max_dia_mm < min_dia_mm:
        st.warning("Max Dia קטן מ-Min Dia — מחליף ביניהם אוטומטית.")
        min_dia_mm, max_dia_mm = max_dia_mm, min_dia_mm

    st.header("4) עיבוד תמונה")
    brightness = st.slider("Brightness", -100, 100, 0, 1)
    contrast   = st.slider("Contrast", 0.5, 2.5, 1.0, 0.05)
    gamma_val  = st.slider("Gamma", 0.5, 2.5, 1.0, 0.05)
    clahe_clip = st.slider("CLAHE Clip", 0.0, 4.0, 0.0, 0.1)
    clahe_tile = st.slider("CLAHE Tile", 2, 32, 8, 2)
    blur_sigma = st.slider("Blur σ", 0.0, 5.0, 0.0, 0.1)
    sharpen_amt   = st.slider("Sharpen Amount", 0.0, 3.0, 0.0, 0.1)
    sharpen_sigma = st.slider("Sharpen σ", 0.3, 3.0, 1.0, 0.1)

    st.header("5) GCODE Settings")
    servo_up = st.number_input("Servo S_up", 0, 255, 30, 1)
    servo_down = st.number_input("Servo S_dn", 0, 255, 90, 1)
    dwell_up = st.number_input("Dwell Up (ms)", 0, 5000, 80, 10)
    dwell_down = st.number_input("Dwell Down (ms)", 0, 5000, 120, 10)
    feedrate = st.number_input("Feedrate XY (mm/min)", 100, 60000, 3000, 100)

# ======== עיבוד ויצירה ========
if up is not None:
    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("לא הצלחתי לקרוא את התמונה. נסה/י קובץ אחר.")
        st.stop()

    # מקור + גרייסקייל
    gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # התאמת גודל פיזי
    img_fit_bgr  = resize_to_physical_dimensions_keep_aspect(img_bgr, WIDTH_MM, HEIGHT_MM, dpi)
    gray_fit     = resize_to_physical_dimensions_keep_aspect(gray_src, WIDTH_MM, HEIGHT_MM, dpi)

    # עיבוד גרייסקייל
    proc = adjust_brightness_contrast(gray_fit, brightness=brightness, contrast=contrast)
    proc = adjust_gamma(proc, gamma=gamma_val)
    if clahe_clip > 0:
        proc = apply_CLAHE(proc, clip=clahe_clip, tile=clahe_tile)
    if blur_sigma > 0:
        proc = gaussian_blur(proc, sigma=blur_sigma)
    if sharpen_amt > 0:
        proc = unsharp_mask(proc, amount=sharpen_amt, sigma=max(0.3, sharpen_sigma))

    # יצירת Stipple
    seed = 42 if fix_seed else None
    stipple_img, points_mm = stipple_layer(proc, dpi, cell_size_mm, max_dots,
                                           sensitivity, min_dia_mm, max_dia_mm,
                                           seed, flip_y)

    # ===== תצוגה =====
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.subheader("מקור")
        st.image(np_bgr_to_pil_rgb(img_fit_bgr), use_container_width=True)
    with c2:
        st.subheader("גרייסקייל (מעובד)")
        st.image(np_gray_to_pil(proc), use_container_width=True)
    with c3:
        st.subheader(f"Stipple — נקודות: {len(points_mm)}")
        st.image(np_gray_to_pil(stipple_img), use_container_width=True)

    st.divider()

    # ===== ייצוא =====
    st.subheader("📤 ייצוא")
    fname_svg, svg_bytes = build_svg(points_mm, WIDTH_MM, HEIGHT_MM)
    fname_csv, csv_bytes = build_csv(points_mm)
    fname_gcode, gcode_bytes = build_gcode(points_mm, servo_up, servo_down, dwell_up, dwell_down, feedrate)

    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button("⬇️ הורדת SVG", data=svg_bytes, file_name=fname_svg, mime="image/svg+xml")
    with colB:
        st.download_button("⬇️ הורדת CSV", data=csv_bytes, file_name=fname_csv, mime="text/csv")
    with colC:
        st.download_button("⬇️ הורדת GCODE", data=gcode_bytes, file_name=fname_gcode, mime="text/plain")
else:
    st.info("העלה/י תמונה בסרגל הצד כדי להתחיל.")
