import io, csv, datetime, random
import numpy as np
import cv2
from PIL import Image

WIDTH_MM, HEIGHT_MM = 700, 500   # משטח עבודה במ״מ

def resize_to_physical_dimensions_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
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
            frac = mean_val / 255.0
            num_dots = int(max_dots * (frac ** sensitivity))
            ch, cw = cell.shape[:2]
            for _ in range(num_dots):
                rx = random.randint(0, max(0, cw - 1))
                ry = random.randint(0, max(0, ch - 1))
                cx, cy = x + rx, y + ry
                if 0 <= cx < w and 0 <= cy < h:
                    intensity = gray[cy, cx] / 255.0
                    dia_mm = min_dia_mm + intensity * (max_dia_mm - min_dia_mm)
                    radius_px = max(1, int((dpi * dia_mm) / 2))
                    cv2.circle(stipple_img, (cx, cy), radius_px, (255,), -1)
                    px_mm, py_mm = cx / dpi, cy / dpi
                    if flip_y:
                        py_mm = HEIGHT_MM - py_mm
                    points_mm.append((px_mm, py_mm, dia_mm))
    return stipple_img, points_mm

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

def make_preview_panel(img_bgr_fit, gray_proc, stipple_img):
    rgb = cv2.cvtColor(img_bgr_fit, cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(gray_proc, cv2.COLOR_GRAY2RGB)
    stip_rgb = cv2.cvtColor(stipple_img, cv2.COLOR_GRAY2RGB)
    panel = np.hstack([rgb, gray_rgb, stip_rgb])
    pil = Image.fromarray(panel)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def run_stipple(
    bgr, dpi=5, cell_size_mm=3, max_dots=60, sensitivity=2.0,
    min_dia_mm=0.2, max_dia_mm=1.5, brightness=0, contrast=1.0, gamma_val=1.0,
    blur_sigma=0.0, sharpen_amt=0.0, sharpen_sigma=1.0, clahe_clip=0.0, clahe_tile=8,
    flip_y=False, fix_seed=True
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_fit_bgr  = resize_to_physical_dimensions_keep_aspect(bgr, WIDTH_MM, HEIGHT_MM, dpi)
    gray_fit     = resize_to_physical_dimensions_keep_aspect(gray, WIDTH_MM, HEIGHT_MM, dpi)
    proc = adjust_brightness_contrast(gray_fit, brightness=brightness, contrast=contrast)
    proc = adjust_gamma(proc, gamma=gamma_val)
    if clahe_clip > 0: proc = apply_CLAHE(proc, clip=clahe_clip, tile=clahe_tile)
    if blur_sigma > 0: proc = gaussian_blur(proc, sigma=blur_sigma)
    if sharpen_amt > 0: proc = unsharp_mask(proc, amount=sharpen_amt, sigma=max(0.3, sharpen_sigma))
    seed = 42 if fix_seed else None
    stipple_img, points_mm = stipple_layer(proc, dpi, cell_size_mm, max_dots,
                                           sensitivity, min_dia_mm, max_dia_mm,
                                           seed, flip_y)
    preview_png_bytes = make_preview_panel(img_fit_bgr, proc, stipple_img)
    return preview_png_bytes, points_mm

def export_svg_bytes(points_mm):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n'
    bg = f'  <rect x="0" y="0" width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n'
    circles = [f'  <circle cx="{x:.3f}" cy="{y:.3f}" r="{dia/2:.3f}" fill="white"/>\n' for (x,y,dia) in points_mm]
    footer = '</svg>\n'
    return fname, (header + bg + "".join(circles) + footer).encode("utf-8")

def export_csv_bytes(points_mm):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.csv"
    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")
    w.writerow(["X_mm","Y_mm","Dia_mm"])
    for (x,y,dia) in points_mm:
        w.writerow([f"{x:.3f}", f"{y:.3f}", f"{dia:.3f}"])
    return fname, sio.getvalue().encode("utf-8")

def export_gcode_bytes(points_mm, servo_up=30, servo_down=90, dwell_up=80, dwell_down=120, feedrate=3000):
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
