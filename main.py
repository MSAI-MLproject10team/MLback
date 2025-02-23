from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import shutil
import os
import uvicorn

from fin2 import process_images
from colclass import get_personal_color  # 수정된 import

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

@app.get("/")
async def root():
    return HTMLResponse(content="<h1>Hello World</h1><p>Welcome to the Image Analysis API</p>")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # 확장자 검사
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=430, 
            detail=f"Invalid file type: {file_ext}. Only PNG, JPG, JPEG are allowed."
        )
    
    # 1. 이미지 업로드
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. 이미지 분석하여 hex color codes 획득
        hex_colors = process_images(file_path)
        
        # 3. hex color codes를 personal colors로 변환
        personal_colors = get_personal_color(hex_colors)
        
        # 4. 결과 반환
        return JSONResponse(content=personal_colors)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)