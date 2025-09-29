import streamlit as st
import numpy as np
import cv2, random, datetime, csv, base64
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -----------------------
# הגדרות כלליות
# -----------------------
WIDTH_MM, HEIGHT_MM = 700, 500
DOT_DIAMETER_MM = 1.0
USE_BASE64_BG = True   # 👈 שנה ל-True אם אופציית NumPy לא עובדת ב-Streamlit Cloud

# -----------------------
# פונקציות עזר
# -----------------------
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
    img=np.zeros((h,w),dtype=np.uint8)
    r=int(dpi*0.5)
    for (x_mm,y_mm) in points:
        x=int(x_mm*dpi); y=int(y_mm*dpi)
        if 0<=x<w and 0<=y<h:
            cv2.circle(img,(x,y),r,255,-1)
    return img

def pil_to_data_url(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64

def export_svg(points):
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn=f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    with open(fn,"w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'<rect width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n')
        for (x,y) in points:
            f.write(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="0.5" fill="white"/>\n')
        f.write("</svg>")
    return fn

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art – עם ציור/מחיקה")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dpi = st.sidebar.slider("DPI",2,20,5)
    cell=st.sidebar.slider("Cell Size mm",1,15,3)
    maxdots=st.sidebar.slider("Max dots",1,80,15)
    sens=st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)

    gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
    auto_points=stipple(gray_fit,dpi,cell,maxdots,sens,seed=42)
    stipple_img=raster_from_points(auto_points,dpi,gray_fit.shape[0],gray_fit.shape[1])

    # רקע לקנבס
    bg_rgb=np.dstack([stipple_img,stipple_img,stipple_img]).astype(np.uint8)

    if USE_BASE64_BG:
        bg_pil = Image.fromarray(bg_rgb,"RGB")
        bg_source = pil_to_data_url(bg_pil)
    else:
        bg_source = bg_rgb

    st.subheader("🖌️ קנבס אינטראקטיבי")
    canvas_res = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=3,
        stroke_color="white",
        background_color="black",
        background_image=bg_source,   # 👈 כאן נבחר המקור
        update_streamlit=True,
        width=stipple_img.shape[1],
        height=stipple_img.shape[0],
        drawing_mode=st.selectbox("מצב ציור",["transform","circle","freedraw","erase"]),
        key="canvas"
    )

    st.subheader("📥 ייצוא")
    if st.button("Export SVG"):
        fn=export_svg(auto_points)
        st.download_button("Download SVG",open(fn,"rb"),file_name=fn)

