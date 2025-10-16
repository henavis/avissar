from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import io, cv2, numpy as np
from stipple_core import run_stipple, export_svg_bytes, export_csv_bytes, export_gcode_bytes

app = FastAPI(title="Stipple Art Web")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html", "r", encoding="utf-8").read()

@app.post("/process")
def process(image: UploadFile = File(...), dpi: int = Form(5), cell_size_mm: int = Form(3),
            max_dots: int = Form(60), sensitivity: float = Form(2.0),
            min_dia_mm: float = Form(0.2), max_dia_mm: float = Form(1.5),
            brightness: int = Form(0), contrast: float = Form(1.0), gamma_val: float = Form(1.0),
            blur_sigma: float = Form(0.0), sharpen_amt: float = Form(0.0),
            sharpen_sigma: float = Form(1.0), clahe_clip: float = Form(0.0),
            clahe_tile: int = Form(8), flip_y: bool = Form(False), fix_seed: bool = Form(True)):
    data = np.frombuffer(image.file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    png_bytes, _ = run_stipple(bgr, dpi, cell_size_mm, max_dots, sensitivity,
                               min_dia_mm, max_dia_mm, brightness, contrast, gamma_val,
                               blur_sigma, sharpen_amt, sharpen_sigma, clahe_clip, clahe_tile,
                               flip_y, fix_seed)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

@app.post("/export/{fmt}")
def export(fmt: str, image: UploadFile = File(...),
           dpi: int = Form(5), cell_size_mm: int = Form(3),
           max_dots: int = Form(60), sensitivity: float = Form(2.0),
           min_dia_mm: float = Form(0.2), max_dia_mm: float = Form(1.5),
           brightness: int = Form(0), contrast: float = Form(1.0), gamma_val: float = Form(1.0),
           blur_sigma: float = Form(0.0), sharpen_amt: float = Form(0.0),
           sharpen_sigma: float = Form(1.0), clahe_clip: float = Form(0.0),
           clahe_tile: int = Form(8), flip_y: bool = Form(False), fix_seed: bool = Form(True),
           servo_up: int = Form(30), servo_down: int = Form(90),
           dwell_up: int = Form(80), dwell_down: int = Form(120),
           feedrate: int = Form(3000)):
    data = np.frombuffer(image.file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    _, points = run_stipple(bgr, dpi, cell_size_mm, max_dots, sensitivity,
                            min_dia_mm, max_dia_mm, brightness, contrast, gamma_val,
                            blur_sigma, sharpen_amt, sharpen_sigma, clahe_clip, clahe_tile,
                            flip_y, fix_seed)
    if fmt == "svg":
        fname, data = export_svg_bytes(points)
        mt = "image/svg+xml"
    elif fmt == "csv":
        fname, data = export_csv_bytes(points)
        mt = "text/csv"
    elif fmt == "gcode":
        fname, data = export_gcode_bytes(points, servo_up, servo_down, dwell_up, dwell_down, feedrate)
        mt = "text/plain"
    else:
        return {"error": "format not supported"}
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(io.BytesIO(data), media_type=mt, headers=headers)
