import streamlit as st
import numpy as np
import cv2, random, datetime, csv
from streamlit_drawable_canvas import st_canvas

WIDTH_MM, HEIGHT_MM = 700, 500

# ===== פונקציות עזר =====
def resize_keep_aspect(image, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, dpi=5):
    target_w_px, target_h_px = int(width_mm*dpi), int(height_mm*dpi)
    ih, iw = image.shape[:2]
    aspect_img = iw/ih
    aspect_target = target_w_px/target_h_px
    if aspect_img > aspect_target:
        new_w, new_h = target_w_px, int(target_w_px/aspect_img)
    else:
        new_h, new_w = target_h_px, int(target_h_px*aspect_img)
    resized = cv2.resize(image,(new_w,new_h))
    canvas = np.zeros((target_h_px,target_w_px),dtype=image.dtype)
    y_off=(target_h_px-new_h)//2; x_off=(target_w_px-new_w)//2
    canvas[y_off:y_off+new_h,x_off:x_off+new_w]=resized
    return canvas

def stipple(gray, dpi=5, cell_size_mm=3, max_dots=15, sens=1.0, seed=None):
    if seed: random.seed(seed)
    h,w=gray.shape; pts=[]
    cell_px=int(cell_size_mm*dpi)
    for y in range(0,h,cell_px):
        for x in range(0,w,cell_px):
            cell=gray[y:y+cell_px,x:x+cell_px]
            if cell.size==0: continue
            mean_val=np.mean(cell)
            frac=mean_val/255
            nd=int(max_dots*(frac**sens))
            ch,cw=cell.shape[:2]
            for _ in range(nd):
                rx,ry=random.randint(0,max(0,cw-1)),random.randint(0,max(0,ch-1))
                pts.append(((x+rx)/dpi,(y+ry)/dpi))
    return pts

def render_points(points,dpi,h,w):
    img=np.zeros((h,w),dtype=np.uint8)
    r=int(dpi*0.5)
    for (x_mm,y_mm) in points:
        x=int(x_mm*dpi); y=int(y_mm*dpi)
        if 0<=x<w and 0<=y<h:
            cv2.circle(img,(x,y),r,255,-1)
    return img

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

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art – ציור ומחיקה על תצוגה מקדימה")

file = st.file_uploader("📂 העלה תמונה", type=["jpg","jpeg","png"])
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # === Sliders ===
    dpi = st.sidebar.slider("DPI",2,20,5)
    cell=st.sidebar.slider("Cell Size mm",1,15,3)
    maxdots=st.sidebar.slider("Max dots",1,80,15)
    sens=st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)
    fix_seed=st.sidebar.checkbox("Fix Seed",False)

    # === Processing ===
    gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
    auto_points=stipple(gray_fit,dpi,cell,maxdots,sens,seed=42 if fix_seed else None)
    stipple_img=render_points(auto_points,dpi,gray_fit.shape[0],gray_fit.shape[1])

    st.subheader("🔎 תצוגה מקדימה – מרכז המסך")
    canvas_res = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=3,
        stroke_color="white",
        background_color="black",
        background_image=None,
        update_streamlit=True,
        width=stipple_img.shape[1],
        height=stipple_img.shape[0],
        drawing_mode=st.selectbox("מצב ציור",["none","point","erase"]),
        key="canvas"
    )

    # === נקודות ידניות ===
    manual_points=[]
    if canvas_res.json_data is not None:
        for obj in canvas_res.json_data["objects"]:
            if obj["type"]=="circle":
                x=obj["left"]; y=obj["top"]
                manual_points.append((x/dpi,y/dpi))

    all_points=auto_points+manual_points
    st.image(stipple_img, caption="Stipple אוטומטי (רקע בלבד)", clamp=True)

    st.subheader("📥 ייצוא")
    if st.button("Export SVG"):
        fn=export_svg(all_points)
        st.download_button("Download SVG",open(fn,"rb"),file_name=fn)
