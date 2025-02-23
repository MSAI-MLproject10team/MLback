from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import shutil
import os
import uuid

from fin2 import process_images
from colclass import ColorClassifierApp

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

@app.get("/")
async def root():
    return HTMLResponse(content="<h1>Hello World</h1><p>Welcome to the Image Analysis API</p>")

@app.post("/upload-image/")
async def upload_image(file: Optional[UploadFile] = None):

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # 확장자 검사
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=500, detail=f"Invalid file type: {file_ext}. Only PNG, JPG, JPEG are allowed.")
    
    # 1. 이미지 업로드

    # 고유한 파일명 생성 (UUID 사용)
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # # # 2. fin2.py의 process_images 함수를 사용하여 이미지 분석
    # analysis_result = process_images(file_path)
    # print(analysis_result)
    # return JSONResponse(content=analysis_result)

    try:
        analysis_result = process_images(file_path)
        print(analysis_result)
        analysis_result_json= JSONResponse(content=analysis_result)
        return analysis_result_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
