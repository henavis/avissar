import streamlit as st
import numpy as np
import cv2, random, datetime
from streamlit_image_coordinates import streamlit_image_coordinates

# ===== פרמטרים כלליים =====
WIDTH_MM, HEIGHT_MM = 700, 500
DOT_DIAMETER_MM = 1.0

def resize_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
    target_w, target_h = int(width_mm*dpi), int(height_mm*dpi)
    ih, iw = image.shape[:2]
    aspect_img = iw / ih
    aspect_target = target_w / target_h
    if aspect_img > aspect_target:
        new_w, new_h = target_w, int(target_w/aspect_img)
    else:
        new_h, new_w = target_h, int(target_h*aspect_img)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    y_off = (target_h - new_h)//2; x_off = (target_w - new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# --- עיבודי תמונה ---
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
    return cv2.GaussianBlur(img, (k,k), sigma)

def apply_sharpen(img, amount=0.0, sigma=1.0):
    if amount <= 0: return img
    blur = apply_blur(img, sigma)
    sharp = img.astype(np.float32) + amount*(img.astype(np.float32)-blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

# --- יצירת נקודות Stipple ---
def stipple(gray, dpi=5, cell_size_mm=3, max_dots=15, sens=1.0, seed=None):
    if seed: random.seed(seed)
    h,w=gray.shape; pts=[]
    cell_px=int(cell_size_mm*dpi)
    for y in range(0,h,cell_px):
        for x in range(0,w,cell_px):
            cell=gray[y:y+cell_px,x:x+cell_px]
            if cell.size==0: continue
            mean_val=np.mean(cell); frac=mean_val/255
            nd=int(max_dots*(frac**sens))
            ch,cw=cell.shape
            for _ in range(nd):
                rx,ry=random.randint(0,cw-1),random.randint(0,ch-1)
                pts.append(((x+rx)/dpi,(y+ry)/dpi))
    return pts

# --- רסטר מתצוגה ---
def raster_from_points(points, dpi, h, w):
    img = np.zeros((h,w), dtype=np.uint8)
    r = max(1, int(dpi*DOT_DIAMETER_MM/2))
    for (x,y) in points:
        cx, cy = int(x*dpi), int(y*dpi)
        if 0<=cx<w and 0<=cy<h:
            cv2.circle(img, (cx,cy), r, 255, -1)
    return img

# --- ייצוא SVG ---
def export_svg(points):
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn=f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    with open(fn,"w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'<rect width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n')
        for (x,y) in points:
            f.write(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{DOT_DIAMETER_MM/2}" fill="white"/>\n')
        f.write("</svg>")
    return fn

# ========================
# Streamlit UI
# ========================
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art – הוספה/מחיקה עם קליק")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if not file:
    st.stop()

file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ===== Sidebar =====
st.sidebar.header("📐 פרמטרי Stipple")
dpi = st.sidebar.slider("DPI",2,20,5)
cell=st.sidebar.slider("Cell Size mm",1,15,3)
maxdots=st.sidebar.slider("Max dots",1,80,15)
sens=st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)

st.sidebar.header("🎚️ עיבודי תמונה")
brightness = st.sidebar.slider("Brightness",-100,100,0)
contrast = st.sidebar.slider("Contrast",0.5,2.0,1.0,0.05)
gamma_val = st.sidebar.slider("Gamma",0.5,2.5,1.0,0.05)
blur_sigma = st.sidebar.slider("Blur σ",0.0,5.0,0.0,0.1)
sharpen_amt = st.sidebar.slider("Sharpen",0.0,2.0,0.0,0.1)

mode = st.sidebar.radio("מצב",["Add","Erase"])

# ===== עיבוד תמונה =====
gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
proc = adjust_brightness_contrast(gray_fit, brightness, contrast)
proc = adjust_gamma(proc, gamma_val)
if blur_sigma>0: proc=apply_blur(proc, blur_sigma)
if sharpen_amt>0: proc=apply_sharpen(proc, sharpen_amt, sigma=1.0)

# ===== נקודות אוטומטיות =====
auto_points=stipple(proc,dpi,cell,maxdots,sens,seed=42)

# === Session state לנקודות ידניות ===
if "manual_points" not in st.session_state:
    st.session_state.manual_points=[]

# ===== Preview =====
all_points = auto_points + st.session_state.manual_points
preview_img = raster_from_points(all_points, dpi, proc.shape[0], proc.shape[1])

# הצגת תמונה עם אפשרות קליק
coords = streamlit_image_coordinates(preview_img, key="clickable_preview")

if coords is not None:
    cx_mm, cy_mm = coords["x"]/dpi, coords["y"]/dpi
    if mode=="Add":
        st.session_state.manual_points.append((cx_mm, cy_mm))
    elif mode=="Erase":
        st.session_state.manual_points = [
            (x,y) for (x,y) in st.session_state.manual_points
            if (x-cx_mm)**2+(y-cy_mm)**2 > (DOT_DIAMETER_MM**2)
        ]

# עדכון אחרי פעולה
all_points = auto_points + st.session_state.manual_points
preview_img = raster_from_points(all_points, dpi, proc.shape[0], proc.shape[1])
st.image(preview_img, caption="Preview עם קליקים", clamp=True, use_column_width=True)

st.caption(f"Auto: {len(auto_points)} | Manual: {len(st.session_state.manual_points)} | Total: {len(all_points)}")

# ===== ייצוא =====
if st.button("Export SVG"):
    fn=export_svg(all_points)
    st.download_button("Download SVG", open(fn,"rb"), file_name=fn)
