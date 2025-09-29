pip install streamlit opencv-python-headless numpy pillow streamlit-drawable-canvas
import streamlit as st
import numpy as np
import cv2, random, datetime, csv, math
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ======================
# פיזיקה של הדף / פלוטר
# ======================
WIDTH_MM, HEIGHT_MM = 700, 500   # גודל העבודה במ"מ (קבוע)
DOT_DIAMETER_MM = 1.0            # קוטר נקודה קבוע 1 מ"מ

# ----------------------
# עיבודי תמונה בסיסיים
# ----------------------
def apply_brightness_contrast(img_gray, brightness=0, contrast=1.0):
    # brightness: [-100..100] (intensities), contrast: [0.5..2.0]
    res = img_gray.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(res, 0, 255).astype(np.uint8)

def apply_gamma(img_gray, gamma=1.0):
    if gamma <= 0: gamma = 1.0
    inv = 1.0 / gamma
    norm = img_gray.astype(np.float32) / 255.0
    out = np.power(norm, inv) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_blur(img_gray, sigma=0.0):
    if sigma <= 0: 
        return img_gray
    # קובע גודל ליבה מתאים לסיגמא
    k = int(max(3, 2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img_gray, (k, k), sigmaX=sigma, sigmaY=sigma)

def apply_unsharp(img_gray, amount=0.0, sigma=1.0):
    if amount <= 0: 
        return img_gray
    blur = apply_blur(img_gray, sigma=max(0.5, sigma))
    us = img_gray.astype(np.float32) + amount * (img_gray.astype(np.float32) - blur.astype(np.float32))
    return np.clip(us, 0, 255).astype(np.uint8)

# ------------------------------
# שינוי גודל למיטת העבודה (mm)
# ------------------------------
def resize_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
    """מתאים את התמונה לגודל המבוקש במ"מ תוך שמירת יחס (padding שחור)."""
    target_w, target_h = int(width_mm * dpi), int(height_mm * dpi)
    ih, iw = image.shape[:2]
    aspect_img = iw / ih
    aspect_target = target_w / target_h

    if aspect_img > aspect_target:
        new_w, new_h = target_w, int(target_w / aspect_img)
    else:
        new_h, new_w = target_h, int(target_h * aspect_img)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(image.shape) == 3:
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ------------------------------------------
# יצירת נקודות אוטומטיות (stipple continuous)
# ------------------------------------------
def generate_stipple_points(gray_fit, dpi=5, cell_mm=3, max_dots=15, sensitivity=1.0, seed=None):
    """מחזיר רשימת נקודות במילימטרים (x_mm, y_mm)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

    h, w = gray_fit.shape
    cell_px = max(1, int(cell_mm * dpi))
    pts = []
    for y in range(0, h, cell_px):
        for x in range(0, w, cell_px):
            ch = min(cell_px, h - y)
            cw = min(cell_px, w - x)
            if ch <= 0 or cw <= 0:
                continue
            cell = gray_fit[y:y+ch, x:x+cw]
            mean_val = float(np.mean(cell))
            frac = mean_val / 255.0                     # בהיר -> הרבה נקודות
            nd = int(max_dots * (frac ** sensitivity))  # רגישות

            for _ in range(nd):
                rx = np.random.randint(0, max(1, cw))
                ry = np.random.randint(0, max(1, ch))
                cx, cy = x + rx, y + ry
                pts.append((cx / dpi, cy / dpi))
    return pts

# ----------------------------
# ציור נקודות לרסטר תצוגה
# ----------------------------
def raster_from_points(points_mm, dpi, h_px, w_px, flip_y=False):
    img = np.zeros((h_px, w_px), dtype=np.uint8)
    r_px = max(1, int(round(dpi * (DOT_DIAMETER_MM / 2.0))))  # רדיוס ב־px
    for (x_mm, y_mm) in points_mm:
        x = int(round(x_mm * dpi))
        y = int(round((HEIGHT_MM - y_mm) * dpi)) if flip_y else int(round(y_mm * dpi))
        if 0 <= x < w_px and 0 <= y < h_px:
            cv2.circle(img, (x, y), r_px, 255, -1)
    return img

# -----------------------------------
# סינון נקודות לפי מסיכות מחיקה
# -----------------------------------
def apply_erase_masks(points_mm, masks):
    if not masks:
        return points_mm
    out = []
    for (x, y) in points_mm:
        keep = True
        for (cx, cy, r_mm) in masks:
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r_mm ** 2):
                keep = False
                break
        if keep:
            out.append((x, y))
    return out

# ------------------------------------------------------
# חילוץ נקודות/מסיכות מתוך JSON של ה-canvas (בדוק היטב)
# ------------------------------------------------------
def extract_dots_and_masks(canvas_json, dpi, erase_color="rgba(255, 0, 0, 1)", brush_mm=2.0):
    """מחזיר (manual_points_mm, erase_masks_mm)."""
    manual = []
    masks = []
    if not canvas_json or "objects" not in canvas_json:
        return manual, masks

    for obj in canvas_json["objects"]:
        typ = obj.get("type", "")
        stroke = obj.get("stroke", obj.get("strokeColor", ""))
        fill = obj.get("fill", obj.get("fillColor", ""))

        # ---- נקודה מסוג circle ----
        if typ == "circle":
            # ב-Fabric: left/top הם פינת מלבן החוסם; המרכז = left+radius*scaleX, top+radius*scaleY
            rad = float(obj.get("radius", obj.get("rx", obj.get("width", 1)/2)))
            sx  = float(obj.get("scaleX", 1.0))
            sy  = float(obj.get("scaleY", 1.0))
            left = float(obj.get("left", 0.0))
            top  = float(obj.get("top", 0.0))
            cx_px = left + rad * sx
            cy_px = top  + rad * sy
            cx_mm, cy_mm = cx_px / dpi, cy_px / dpi

            # אם זו "מחיקה" נסמן כמסיכה (צבע אדום), אחרת נקודה לבנה ידנית
            if stroke == erase_color or fill == erase_color:
                # רדיוס מסכה מתוך גודל הציור בפיקסלים -> מ"מ
                r_mm = (rad * sx) / dpi
                r_mm = max(r_mm, brush_mm)  # הבטחת מינימום הרדיוס מסליידר
                masks.append((cx_mm, cy_mm, r_mm))
            else:
                manual.append((cx_mm, cy_mm))

        # ---- מסלול חופשי (freedraw / path) ----
        elif typ == "path" and "path" in obj:
            # נתיב מורכב מצעדים ["M"/"L"/... , x, y]
            path = obj["path"]
            color = stroke or fill
            for step in path:
                if isinstance(step, list) and len(step) >= 3:
                    # לרוב step = ["L", x, y] או ["M", x, y]
                    x = step[-2]; y = step[-1]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        x_mm, y_mm = x / dpi, y / dpi
                        if color == erase_color:
                            masks.append((x_mm, y_mm, brush_mm))
                        else:
                            manual.append((x_mm, y_mm))
        # התעלמות מצורות אחרות (rect/line/polygon וכו')
    return manual, masks

# -------------------------
# יצוא SVG / GCODE / CSV
# -------------------------
def export_svg(points, flip_y=False):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'<rect x="0" y="0" width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n')
        for (x, y) in points:
            cy = (HEIGHT_MM - y) if flip_y else y
            f.write(f'<circle cx="{x:.3f}" cy="{cy:.3f}" r="{DOT_DIAMETER_MM/2:.3f}" fill="white"/>\n')
        f.write("</svg>\n")
    return fn

def export_gcode(points, flip_y=False, S_UP=30, S_DN=90, P_UP=80, P_DN=120, F=3000):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.gcode"
    lines = ["(Stipple for GRBL)","G21","G90",f"F{F}",f"M3 S{S_UP}",f"G4 P{P_UP}"]
    for (x, y) in points:
        cy = (HEIGHT_MM - y) if flip_y else y
        lines.append(f"G0 X{x:.3f} Y{cy:.3f}")
        lines.append(f"M3 S{S_DN}")
        lines.append(f"G4 P{P_DN}")
        lines.append(f"M3 S{S_UP}")
        lines.append(f"G4 P{P_UP}")
    lines += ["G0 X0 Y0","M5","(End)"]
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return fn

def export_csv(points, flip_y=False):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["X_mm","Y_mm"])
        for (x, y) in points:
            cy = (HEIGHT_MM - y) if flip_y else y
            w.writerow([f"{x:.3f}", f"{cy:.3f}"])
    return fn

# ======================
# Streamlit UI
# ======================
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art – תצוגה מרכזית עם ציור/מחיקה + עיבודי תמונה")

# העלאת תמונה
file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if not file:
    st.info("העלה תמונה כדי להתחיל.")
    st.stop()

# קריאה וגרייסקייל
file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
if img_bgr is None:
    st.error("לא ניתן לקרוא את הקובץ. נסה תמונה אחרת.")
    st.stop()
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ===== Sidebar – שליטה =====
st.sidebar.header("⚙️ הגדרות")
dpi = st.sidebar.slider("DPI (px/mm)", 2, 20, 5)
cell_mm = st.sidebar.slider("Cell Size (mm)", 1, 15, 3)
max_dots = st.sidebar.slider("Max dots / cell", 1, 80, 15)
sensitivity = st.sidebar.slider("Sensitivity", 0.2, 4.0, 1.0, 0.1)
flip_y = st.sidebar.checkbox("Flip Y (לפלוטר)", False)
fix_seed = st.sidebar.checkbox("Fix Random Seed", False)

st.sidebar.header("🎚️ עיבודי תמונה")
brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast   = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
gamma_val  = st.sidebar.slider("Gamma", 0.5, 2.5, 1.0, 0.05)
blur_sigma = st.sidebar.slider("Blur σ", 0.0, 5.0, 0.0, 0.1)
sharpen_amt= st.sidebar.slider("Sharpen amount", 0.0, 2.0, 0.0, 0.1)

st.sidebar.header("🖌️ ציור/מחיקה")
draw_mode = st.sidebar.selectbox("מצב על הקנבס", ["הוסף נקודות (עיגולים)", "מחיקה (אדום)", "Transform/None"])
dot_visual_mm = st.sidebar.slider("גודל עיגול לציור (ויזואלי בלבד, מ\"מ)", 0.5, 5.0, 1.0, 0.5)
erase_brush_mm = st.sidebar.slider("מברשת מחיקה (מ\"מ)", 0.5, 20.0, 3.0, 0.5)

preview_width = st.sidebar.slider("Preview Width (px)", 700, 1600, 1100, 50)

st.sidebar.header("🧰 ייצוא – G-code")
S_UP = st.sidebar.number_input("Servo S_up", 0, 255, 30)
S_DN = st.sidebar.number_input("Servo S_dn", 0, 255, 90)
P_UP = st.sidebar.number_input("Dwell Up (ms)", 0, 5000, 80, 10)
P_DN = st.sidebar.number_input("Dwell Down (ms)", 0, 5000, 120, 10)
F_XY = st.sidebar.number_input("Feedrate F (mm/min)", 100, 30000, 3000, 100)

# ===== עיבוד גרייסקייל על פי הסליידרים =====
gray_fit_raw = resize_keep_aspect(gray_src, WIDTH_MM, HEIGHT_MM, dpi)
gray_proc = apply_brightness_contrast(gray_fit_raw, brightness=brightness, contrast=contrast)
gray_proc = apply_gamma(gray_proc, gamma=gamma_val)
if blur_sigma > 0:
    gray_proc = apply_blur(gray_proc, sigma=blur_sigma)
if sharpen_amt > 0:
    gray_proc = apply_unsharp(gray_proc, amount=sharpen_amt, sigma=max(0.8, blur_sigma if blur_sigma>0 else 1.0))

# ===== נקודות אוטומטיות + רסטר תצוגה =====
seed_val = 42 if fix_seed else None
auto_points = generate_stipple_points(gray_proc, dpi=dpi, cell_mm=cell_mm, max_dots=max_dots, sensitivity=sensitivity, seed=seed_val)
stipple_img = raster_from_points(auto_points, dpi, gray_proc.shape[0], gray_proc.shape[1], flip_y=False)  # התצוגה תמיד y רגיל

# ===== תצוגה: מקור + גרייסקייל =====
st.subheader("תצוגות עזר")
c1, c2 = st.columns(2)
with c1:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="תמונה מקורית", use_column_width=True)
with c2:
    st.image(gray_proc, caption="תמונה בגווני אפור (לאחר עיבודים)", clamp=True, use_column_width=True)

# ===== תצוגה מרכזית – קנבס לציור/מחיקה =====
st.subheader("תצוגה מרכזית: Stipple + ציור/מחיקה")
# רקע הקנבס = תמונת ה-stipple (שחור/לבן)
bg_rgb = np.dstack([stipple_img, stipple_img, stipple_img])  # 1ch->3ch
bg_rgb = bg_rgb.astype(np.uint8)
bg_pil = Image.fromarray(bg_rgb, mode="RGB")

# הגדרת מצב קנבס
if draw_mode == "הוסף נקודות (עיגולים)":
    drawing_mode = "circle"
    stroke_color = "rgba(255, 255, 255, 1)"  # לבן
    fill_color   = "rgba(255, 255, 255, 1)"
elif draw_mode == "מחיקה (אדום)":
    drawing_mode = "circle"   # גם מחיקה כעיגולים אדומים
    stroke_color = "rgba(255, 0, 0, 1)"      # אדום -> יתורגם למסיכות מחיקה
    fill_color   = "rgba(255, 0, 0, 1)"
else:
    drawing_mode = "transform"  # אין ציור
    stroke_color = "rgba(255, 255, 255, 1)"
    fill_color   = "rgba(255, 255, 255, 1)"

# קוטר ויזואלי בלבד (פיקסלים)
dot_visual_px = max(3, int(round(dot_visual_mm * dpi)))
canvas_res = st_canvas(
    fill_color=fill_color,
    stroke_width=dot_visual_px,       # circle radius בקאנבס תלוי width/scale; נשתמש בזה כסיגנל
    stroke_color=stroke_color,
    background_color="black",
    background_image=bg_pil,          # מציירים מעל התצוגה
    update_streamlit=True,
    width=stipple_img.shape[1] * preview_width // stipple_img.shape[1],  # התאמה פשוטה לרוחב
    height=stipple_img.shape[0] * preview_width // stipple_img.shape[1],
    drawing_mode=drawing_mode,
    key="stipple_canvas",
)

# חילוץ נקודות ידניות + מסיכות מחיקה מהקנבס
manual_points, erase_masks = extract_dots_and_masks(
    canvas_res.json_data, dpi=dpi, erase_color="rgba(255, 0, 0, 1)", brush_mm=erase_brush_mm
)

# מסננים מחיקות ומוסיפים נק' ידניות
auto_filtered = apply_erase_masks(auto_points, erase_masks)
manual_filtered = apply_erase_masks(manual_points, erase_masks)
all_points = auto_filtered + manual_filtered

st.caption(f"נקודות אוטומטיות: {len(auto_points)} | ידניות שנוספו: {len(manual_points)} | לאחר מחיקה: {len(all_points)}")

# ===== ייצוא =====
st.subheader("📥 ייצוא")
cA, cB, cC = st.columns(3)
with cA:
    if st.button("Export SVG"):
        fn = export_svg(all_points, flip_y=flip_y)
        st.success(f"נשמר: {fn}")
        st.download_button("Download SVG", open(fn, "rb"), file_name=fn)
with cB:
    if st.button("Export G-code"):
        fn = export_gcode(all_points, flip_y=flip_y, S_UP=S_UP, S_DN=S_DN, P_UP=P_UP, P_DN=P_DN, F=F_XY)
        st.success(f"נשמר: {fn}")
        st.download_button("Download G-code", open(fn, "rb"), file_name=fn)
with cC:
    if st.button("Export CSV"):
        fn = export_csv(all_points, flip_y=flip_y)
        st.success(f"נשמר: {fn}")
        st.download_button("Download CSV", open(fn, "rb"), file_name=fn)
