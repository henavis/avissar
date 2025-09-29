import streamlit as st
import numpy as np
import cv2, random, datetime

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
st.title("🎨 Stipple Art – ציור/מחיקה על Preview")

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

# ===== יצירת נקודות אוטומטיות =====
gray_fit=resize_keep_aspect(gray_src,WIDTH_MM,HEIGHT_MM,dpi)
auto_points=stipple(gray_fit,dpi,cell,maxdots,sens,seed=42)

# === אחסון נקודות ידניות במחסן session_state ===
if "manual_points" not in st.session_state:
    st.session_state.manual_points=[]
if "erase_points" not in st.session_state:
    st.session_state.erase_points=[]

# ===== כפתורים להוספה/מחיקה =====
col1,col2=st.columns(2)
with col1:
    add_x = st.number_input("X (mm) להוספה",0.0,WIDTH_MM,0.0,step=1.0)
    add_y = st.number_input("Y (mm) להוספה",0.0,HEIGHT_MM,0.0,step=1.0)
    if st.button("➕ הוסף נקודה"):
        st.session_state.manual_points.append((add_x,add_y))
with col2:
    del_x = st.number_input("X (mm) למחיקה",0.0,WIDTH_MM,0.0,step=1.0)
    del_y = st.number_input("Y (mm) למחיקה",0.0,HEIGHT_MM,0.0,step=1.0)
    if st.button("❌ מחק נקודה קרובה"):
        st.session_state.erase_points.append((del_x,del_y))

# ===== מיזוג נקודות =====
all_points = auto_points + st.session_state.manual_points

# מחיקה: מסירים נקודות קרובות ל־erase_points
final_points=[]
for (x,y) in all_points:
    keep=True
    for (ex,ey) in st.session_state.erase_points:
        if (x-ex)**2+(y-ey)**2 < (DOT_DIAMETER_MM**2):
            keep=False; break
    if keep: final_points.append((x,y))

# ===== Preview =====
preview_img = raster_from_points(final_points, dpi, gray_fit.shape[0], gray_fit.shape[1])
st.image(preview_img, caption="Preview עם ציור/מחיקה", clamp=True, use_column_width=True)

st.caption(f"Auto: {len(auto_points)} | Manual: {len(st.session_state.manual_points)} | After erase: {len(final_points)}")

# ===== ייצוא =====
if st.button("Export SVG"):
    fn=export_svg(final_points)
    st.download_button("Download SVG", open(fn,"rb"), file_name=fn)
