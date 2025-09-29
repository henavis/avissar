import streamlit as st
import numpy as np
import cv2, random, datetime

# ===== הגדרות כלליות =====
WIDTH_MM, HEIGHT_MM = 700, 500
DOT_DIAMETER_MM = 1.0  # קוטר נקודה קבוע 1 מ"מ

# --- שמירת יחס בעת שינוי גודל (מ"מ -> פיקסלים) ---
def resize_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
    target_w, target_h = int(width_mm*dpi), int(height_mm*dpi)
    ih, iw = image.shape[:2]
    aspect_img = iw / ih
    aspect_target = target_w / target_h
    if aspect_img > aspect_target:
        new_w, new_h = target_w, int(target_w/aspect_img)
    else:
        new_h, new_w = target_h, int(target_h*aspect_img)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(image.shape) == 3:
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    y_off = (target_h - new_h)//2
    x_off = (target_w - new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# --- עיבודי תמונה בסיסיים ---
def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    res = img.astype(np.float32) * contrast + brightness
    return np.clip(res, 0, 255).astype(np.uint8)

def adjust_gamma(img, gamma=1.0):
    inv = 1.0 / max(gamma, 0.01)
    norm = img.astype(np.float32) / 255.0
    return np.clip((norm**inv)*255, 0, 255).astype(np.uint8)

def apply_blur(img, sigma=0.0):
    if sigma <= 0: return img
    k = int(max(3, 2*round(3*sigma)+1))
    return cv2.GaussianBlur(img, (k,k), sigmaX=sigma, sigmaY=sigma)

def apply_sharpen(img, amount=0.0, sigma=1.0):
    if amount <= 0: return img
    blur = apply_blur(img, sigma=max(0.5, sigma))
    sharp = img.astype(np.float32) + amount*(img.astype(np.float32)-blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

# --- יצירת נקודות Stipple (אזורים בהירים -> יותר נקודות) ---
def stipple(gray, dpi=5, cell_size_mm=3, max_dots=15, sens=1.0, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
    h, w = gray.shape
    pts = []
    cell_px = max(1, int(cell_size_mm*dpi))
    for y in range(0, h, cell_px):
        for x in range(0, w, cell_px):
            ch = min(cell_px, h - y)
            cw = min(cell_px, w - x)
            if ch <= 0 or cw <= 0:
                continue
            cell = gray[y:y+ch, x:x+cw]
            frac = float(np.mean(cell))/255.0
            nd = int(max_dots * (frac**sens))
            for _ in range(nd):
                rx = np.random.randint(0, cw)
                ry = np.random.randint(0, ch)
                cx, cy = x + rx, y + ry
                pts.append((cx/dpi, cy/dpi))
    return pts

# --- רסטר תצוגה מנקודות (שחור רקע, נקודות לבנות) ---
def raster_from_points(points, dpi, h, w):
    img = np.zeros((h, w), dtype=np.uint8)
    r = max(1, int(round(dpi * (DOT_DIAMETER_MM/2.0))))
    for (x_mm, y_mm) in points:
        x = int(round(x_mm * dpi))
        y = int(round(y_mm * dpi))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), r, 255, -1)
    return img

# --- יצוא SVG ---
def export_svg(points):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'  <rect width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n')
        for (x, y) in points:
            f.write(f'  <circle cx="{x:.3f}" cy="{y:.3f}" r="{DOT_DIAMETER_MM/2:.3f}" fill="white"/>\n')
        f.write('</svg>\n')
    return fn

# ========================
# Streamlit UI
# ========================
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art – תצוגה גדולה + עיבודי תמונה (ללא ציור/מחיקה)")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if not file:
    st.info("העלה תמונה כדי להתחיל.")
    st.stop()

# קריאת תמונה וגרייסקייל
file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
if img_bgr is None:
    st.error("לא הצלחתי לקרוא את הקובץ. נסה תמונה אחרת.")
    st.stop()
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ===== Sidebar =====
st.sidebar.header("📐 פרמטרי Stipple")
dpi       = st.sidebar.slider("DPI (px/mm)", 2, 20, 5)
cell_mm   = st.sidebar.slider("Cell Size (mm)", 1, 15, 3)
max_dots  = st.sidebar.slider("Max dots per cell", 1, 80, 15)
sensitivity = st.sidebar.slider("Sensitivity", 0.2, 4.0, 1.0, 0.1)
fix_seed  = st.sidebar.checkbox("Fix Random Seed (יציבות)", False)

st.sidebar.header("🎚️ עיבודי תמונה")
brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast   = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
gamma_val  = st.sidebar.slider("Gamma", 0.5, 2.5, 1.0, 0.05)
blur_sigma = st.sidebar.slider("Blur σ", 0.0, 5.0, 0.0, 0.1)
sharpen_amt= st.sidebar.slider("Sharpen amount", 0.0, 2.0, 0.0, 0.1)

# ===== עיבוד וייצור נקודות =====
gray_fit   = resize_keep_aspect(gray_src, WIDTH_MM, HEIGHT_MM, dpi)
proc_gray  = adjust_brightness_contrast(gray_fit, brightness, contrast)
proc_gray  = adjust_gamma(proc_gray, gamma_val)
if blur_sigma > 0:
    proc_gray = apply_blur(proc_gray, blur_sigma)
if sharpen_amt > 0:
    # אם יש טשטוש, נשתמש באותה σ לחידוד עדין; אחרת 1.0
    proc_gray = apply_sharpen(proc_gray, sharpen_amt, sigma=max(0.8, blur_sigma if blur_sigma>0 else 1.0))

seed_val = 42 if fix_seed else None
auto_points = stipple(proc_gray, dpi=dpi, cell_size_mm=cell_mm, max_dots=max_dots, sens=sensitivity, seed=seed_val)
preview_img = raster_from_points(auto_points, dpi, proc_gray.shape[0], proc_gray.shape[1])

# ===== תצוגות: מקור + Grayscale קטן; Stipple גדול =====
st.subheader("תצוגות עזר (קטנות)")
c1, c2 = st.columns(2)
with c1:
    st.image(cv2.cvtColor(resize_keep_aspect(img_bgr, WIDTH_MM, HEIGHT_MM, dpi), cv2.COLOR_BGR2RGB),
             caption="תמונה מקורית (מותאמת)", width=360)
with c2:
    st.image(proc_gray, caption="Grayscale לאחר עיבוד", clamp=True, width=360)

st.subheader("🖼️ תצוגת Stipple גדולה")
st.image(preview_img, caption=f"Stipple Preview — נקודות: {len(auto_points)}", clamp=True, use_column_width=True)

# ===== ייצוא =====
st.subheader("📥 ייצוא")
if st.button("Export SVG"):
    fn = export_svg(auto_points)
    st.success(f"נשמר: {fn}")
    st.download_button("Download SVG", open(fn, "rb"), file_name=fn)
