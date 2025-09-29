import streamlit as st
import numpy as np
import cv2, random, datetime, csv
from streamlit_drawable_canvas import st_canvas

# ===== פרמטרים כלליים =====
WIDTH_MM, HEIGHT_MM = 700, 500
DOT_DIAMETER_MM = 1.0

# --- Resize לשמירה על יחס ---
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

# --- החלת מחיקות ---
def apply_erase_masks(points, masks):
    out=[]
    for (x,y) in points:
        keep=True
        for (cx,cy,r_mm) in masks:
            if (x-cx)**2+(y-cy)**2 <= r_mm**2:
                keep=False; break
        if keep: out.append((x,y))
    return out

# --- יצוא SVG ---
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
st.title("🎨 Stipple Art – עם נקודות ידניות ומחיקה")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if not file:
    st.info("אנא העלה תמונה כדי להתחיל")
    st.stop()

# קריאה לתמונה
file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ===== SideBar =====
dpi = st.sidebar.slider("DPI",2,20,3)
cell=st.sidebar.slider("Cell Size mm",1,15,2)
maxdots=st.sidebar.slider("Max dots",1,80,40)
sens=st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)
erase_brush_mm = st.sidebar.slider("מברשת מחיקה (מ\"מ)",0.5,20.0,3.0,0.5)

draw_mode = st.sidebar.selectbox("מצב ציור",["None","Add Points","Erase"])

# ===== עיבוד =====
gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
auto_points=stipple(gray_fit,dpi,cell,maxdots,sens,seed=42)

# תצוגת גרייסקייל
st.subheader("תצוגה: מקור + Grayscale")
c1,c2 = st.columns(2)
with c1:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="תמונה מקורית", use_column_width=True)
with c2:
    st.image(gray_fit, caption="Grayscale (מותאם)", clamp=True, use_column_width=True)

# ===== קנבס לציור =====
st.subheader("🖌️ הוספת נקודות / מחיקה")
if draw_mode=="Add Points":
    mode="circle"; stroke="rgba(255,255,255,1)"
elif draw_mode=="Erase":
    mode="circle"; stroke="rgba(255,0,0,1)"
else:
    mode="transform"; stroke="rgba(255,255,255,1)"

canvas_res = st_canvas(
    fill_color=stroke,
    stroke_width=int(dpi*DOT_DIAMETER_MM),  # גודל מברשת
    stroke_color=stroke,
    background_color="black",
    update_streamlit=True,
    width=gray_fit.shape[1],
    height=gray_fit.shape[0],
    drawing_mode=mode,
    key="canvas"
)

# ===== חילוץ נקודות ידניות + מסיכות מחיקה =====
manual_points=[]; erase_masks=[]
if canvas_res.json_data is not None:
    for obj in canvas_res.json_data["objects"]:
        if obj["type"]=="circle":
            cx = obj["left"]+obj["radius"]
            cy = obj["top"]+obj["radius"]
            cx_mm, cy_mm = cx/dpi, cy/dpi
            if obj["fill"]=="rgba(255, 0, 0, 1)":   # אדום -> מחיקה
                erase_masks.append((cx_mm,cy_mm,erase_brush_mm))
            else:
                manual_points.append((cx_mm,cy_mm))

# ===== מיזוג =====
auto_filtered = apply_erase_masks(auto_points, erase_masks)
manual_filtered = apply_erase_masks(manual_points, erase_masks)
all_points = auto_filtered + manual_filtered

st.caption(f"נקודות אוטומטיות: {len(auto_points)} | ידניות: {len(manual_points)} | לאחר מחיקה: {len(all_points)}")

# ===== ייצוא =====
st.subheader("📥 ייצוא")
if st.button("Export SVG"):
    fn=export_svg(all_points)
    st.download_button("Download SVG",open(fn,"rb"),file_name=fn)
