from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import shutil
import os
import uvicorn

from new_fin2 import process_images
from origin_files.colclass import get_personal_color  # 수정된 import
from upload_image import upload_new_image
from adjust_white_balance import adjust_white_balance

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
PROCESSED_DIRECTORY = "processed_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png":
        return "image/png"
    elif ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    else:
        return "application/octet-stream"  # 기본값, 필요한 경우 추가 설정 가능


@app.get("/")
async def root():
    return HTMLResponse(content="<h1>Hello World</h1><p>Welcome to the Image Analysis API</p>")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """이미지를 서버에 업로드하고 파일명을 반환"""
    # 1. 이미지 업로드
    file_name = upload_new_image(file, UPLOAD_DIRECTORY)
    return {"filename": file_name}
    

@app.get("/adjust-image/{filename}")
async def adjust_image(filename: str):
    """서버에 저장된 이미지를 보정"""
    original_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(original_path):
        raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")

    try:
        adjusted_path = os.path.join(PROCESSED_DIRECTORY, f"adjusted_{filename}")
        # 2. 화이트 밸런스 맞추기
        adjust_white_balance(original_path, adjusted_path)
        # 미디어 타입 동적으로 지정 # 지정하는 것이 더 명확하고 효율적인 동작이 가능
        media_type = get_media_type(adjusted_path)
        return FileResponse(adjusted_path, media_type=media_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/extract-color/{filename}")
async def extract_color(filename: str):
    """보정된 이미지에서 색상 추출"""
    adjusted_path = os.path.join(PROCESSED_DIRECTORY, f"adjusted_{filename}")
    if not os.path.exists(adjusted_path):
        raise HTTPException(status_code=404, detail="보정된 이미지가 존재하지 않습니다.")

    try:
        # 2. 이미지 분석하여 hex color codes 획득
        hex_colors = process_images(adjusted_path)
        
        # 3. hex color codes를 personal colors로 변환
        personal_colors = get_personal_color(hex_colors)
        
        # 4. 결과 반환
        return JSONResponse(content=personal_colors)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
