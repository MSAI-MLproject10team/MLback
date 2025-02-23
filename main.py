from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from fin2 import process_images
from colclass import ColorClassifierApp

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

@app.get("/")
async def root():
    return HTMLResponse(content="<h1>Hello World</h1><p>Welcome to the Image Analysis API</p>")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):

    # 1. 이미지 업로드
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # # 2. fin2.py의 process_images 함수를 사용하여 이미지 분석
    analysis_result = process_images(file_path)
    return JSONResponse(content=analysis_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
