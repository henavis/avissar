import streamlit as st
import numpy as np
import cv2, random, datetime, csv
from streamlit_drawable_canvas import st_canvas

WIDTH_MM, HEIGHT_MM = 700, 500

# ===== פונקציות עזר =====
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
    canvas = np.zeros((target_h_px, target_w_px), dtype=image.dtype)
    y_off = (target_h_px - new_h) // 2
    x_off = (target_w_px - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def stipple_art_continuous(gray, dpi=5, cell_size_mm=3, max_dots=15, sensitivity=1.0, seed=None, flip_y=False):
    if seed is not None:
        random.seed(seed)
    h, w = gray.shape
    dot_radius_px = max(1, int(dpi/2))
    cell_px = max(1, int(cell_size_mm * dpi))
    stipple_img = np.zeros((h, w), dtype=np.uint8)
    points_mm = []
    for y in range(0, h, cell_px):
        for x in range(0, w, cell_px):
            cell = gray[y:y+cell_px, x:x+cell_px]
            if cell.size == 0:
                continue
            mean_val = np.mean(cell)
            frac = mean_val / 255.0
            num_dots = int(max_dots * (frac ** sensitivity))
            ch, cw = cell.shape[:2]
            for _ in range(num_dots):
                rx = random.randint(0, max(0, cw - 1))
                ry = random.randint(0, max(0, ch - 1))
                cx, cy = x + rx, y + ry
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(stipple_img, (cx, cy), dot_radius_px, (255,), -1)
                    px_mm, py_mm = cx / dpi, cy / dpi
                    if flip_y:
                        py_mm = HEIGHT_MM - py_mm
                    points_mm.append((px_mm, py_mm))
    return stipple_img, points_mm

def export_svg(points):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.svg"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'<rect x="0" y="0" width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="black"/>\n')
        for (x,y) in points:
            f.write(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="0.5" fill="white"/>\n')
        f.write("</svg>")
    return fname

def export_gcode(points, S_UP=30,S_DN=90,P_UP=80,P_DN=120,F=3000):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.gcode"
    lines=["(Stipple export for GRBL)","G21","G90",f"F{F}",f"M3 S{S_UP}","G4 P{}".format(P_UP)]
    for (x,y) in points:
        lines.append(f"G0 X{x:.3f} Y{y:.3f}")
        lines.append(f"M3 S{S_DN}")
        lines.append(f"G4 P{P_DN}")
        lines.append(f"M3 S{S_UP}")
        lines.append(f"G4 P{P_UP}")
    lines += ["G0 X0 Y0","M5","(End)"]
    with open(fname,"w") as f: f.write("\n".join(lines))
    return fname

def export_csv(points):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"stipple_{WIDTH_MM}x{HEIGHT_MM}_{ts}.csv"
    with open(fname,"w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["X_mm","Y_mm"])
        for (x,y) in points:
            writer.writerow([f"{x:.3f}",f"{y:.3f}"])
    return fname

# ===== ממשק Streamlit =====
st.set_page_config(layout="wide")
st.title("🎨 Stipple Art Generator with Manual Edit")

file = st.file_uploader("העלה תמונה", type=["jpg","jpeg","png"])
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    dpi = st.sidebar.slider("DPI (px/mm)",2,20,5)
    cell_size_mm = st.sidebar.slider("Cell Size (mm)",1,15,3)
    max_dots = st.sidebar.slider("Max dots per cell",1,80,15)
    sensitivity = st.sidebar.slider("Sensitivity",0.2,4.0,1.0,0.1)
    flip_y = st.sidebar.checkbox("Flip Y",False)
    fix_seed = st.sidebar.checkbox("Fix Random Seed",False)

    seed = 42 if fix_seed else None
    gray_fit = resize_to_physical_dimensions_keep_aspect(gray_src, WIDTH_MM, HEIGHT_MM, dpi)
    stipple_img, auto_points = stipple_art_continuous(gray_fit,dpi,cell_size_mm,max_dots,sensitivity,seed,flip_y)

    col1,col2,col3 = st.columns(3)
    with col1: st.image(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB), caption="Original")
    with col2: st.image(gray_fit, caption="Gray Fit", clamp=True)
    with col3: st.image(stipple_img, caption="Stipple Auto", clamp=True)

    st.subheader("✏️ עריכה ידנית (הוסף/מחק נקודות)")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=3,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=stipple_img.shape[0],
        width=stipple_img.shape[1],
        drawing_mode=st.selectbox("מצב ציור",["transform","freedraw","erase"]),
        key="canvas"
    )

    manual_points=[]
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"]=="path":
                for (x,y) in obj["path"]:
                    if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                        manual_points.append((x/dpi,y/dpi))

    all_points = auto_points + manual_points

    st.subheader("📥 ייצוא")
    if st.button("Export SVG"):
        fn = export_svg(all_points)
        st.download_button("Download SVG",open(fn,"rb"),file_name=fn)
    if st.button("Export GCODE"):
        fn = export_gcode(all_points)
        st.download_button("Download GCODE",open(fn,"rb"),file_name=fn)
    if st.button("Export CSV"):
        fn = export_csv(all_points)
        st.download_button("Download CSV",open(fn,"rb"),file_name=fn)
