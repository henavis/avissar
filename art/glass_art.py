import streamlit as st
import numpy as np
import cv2, random, datetime, csv
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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

def raster_from_points(points, dpi, h, w):
    img = np.zeros((h,w), dtype=np.uint8)
    r = max(1, int(dpi*DOT_DIAMETER_MM/2))
    for (x,y) in points:
        cx, cy = int(x*dpi), int(y*dpi)
        if 0<=cx<w and 0<=cy<h:
            cv2.circle(img, (cx,cy), r, 255, -1)
    return img

def apply_erase_masks(points, masks):
    out=[]
    for (x,y) in points:
        keep=True
        for (cx,cy,r_mm) in masks:
            if (x-cx)**2+(y-cy)**2 <= r_mm**2:
                keep=False; break
        if keep: out.append((x,y))
    return out

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
st.title("🎨 Stipple Art – Preview אינטראקטיבי")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if not file:
    st.stop()

file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ===== Sidebar =====
dpi = st.sidebar.slider("DPI",2,20,5)
cell=st.sidebar.slider("Cell Size mm",1,15,3)
maxdots=st.sidebar.slider("Max dots",1,80,15)
sens=st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)
erase_brush_mm = st.sidebar.slider("מברשת מחיקה (מ\"מ)",0.5,20.0,3.0,0.5)
draw_mode = st.sidebar.radio("מצב ציור",["Add","Erase","None"])

# ===== עיבוד =====
gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
auto_points=stipple(gray_fit,dpi,cell,maxdots,sens,seed=42)
preview_img = raster_from_points(auto_points, dpi, gray_fit.shape[0], gray_fit.shape[1])

# הופכים את ה-preview לרקע (PIL)
bg_rgb = np.dstack([preview_img, preview_img, preview_img]).astype(np.uint8)
bg_pil = Image.fromarray(bg_rgb)

if draw_mode=="Add":
    mode="circle"; stroke="rgba(255,255,255,1)"
elif draw_mode=="Erase":
    mode="circle"; stroke="rgba(255,0,0,1)"
else:
    mode="transform"; stroke="rgba(255,255,255,1)"

st.subheader("🖌️ תצוגה מקדימה עם ציור/מחיקה")
canvas_res = st_canvas(
    fill_color=stroke,
    stroke_width=int(dpi*DOT_DIAMETER_MM),
    stroke_color=stroke,
    background_color="black",
    background_image=bg_pil,
    update_streamlit=True,
    width=preview_img.shape[1],
    height=preview_img.shape[0],
    drawing_mode=mode,
    key="canvas"
)

# חילוץ נקודות מהקנבס
manual_points=[]; erase_masks=[]
if canvas_res.json_data is not None:
    for obj in canvas_res.json_data["objects"]:
        if obj["type"]=="circle":
            cx=obj["left"]+obj["radius"]
            cy=obj["top"]+obj["radius"]
            cx_mm,cy_mm=cx/dpi,cy/dpi
            if obj["fill"]=="rgba(255, 0, 0, 1)":
                erase_masks.append((cx_mm,cy_mm,erase_brush_mm))
            else:
                manual_points.append((cx_mm,cy_mm))

auto_filtered = apply_erase_masks(auto_points, erase_masks)
manual_filtered = apply_erase_masks(manual_points, erase_masks)
all_points = auto_filtered + manual_filtered

st.caption(f"Auto: {len(auto_points)} | Manual: {len(manual_points)} | After erase: {len(all_points)}")

# ===== ייצוא =====
if st.button("Export SVG"):
    fn=export_svg(all_points)
    st.download_button("Download SVG", open(fn,"rb"), file_name=fn)
