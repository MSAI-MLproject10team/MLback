from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import shutil
import os
import uvicorn

from fin2 import process_images
from colclass import get_personal_color  # 수정된 import
from upload_image import upload_new_image
from adjust_white_balance import adjust_white_balance

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
PROCESSED_DIRECTORY = "processed_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

app.state.new_image_id = ""  # 전역 변수 대신 FastAPI state 사용

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
    file_path, unique_id = upload_new_image(file, UPLOAD_DIRECTORY)
    app.state.new_image_id = unique_id  # FastAPI state에 저장
    # 업로드된 파일 ID 반환
    return JSONResponse(content={"id": unique_id})
    

@app.post("/adjust-image/")
async def adjust_image():
    file_id = app.state.new_image_id
    # 고유 ID를 기준으로 이미지 파일 찾기
    file_path = None

    # 확장자에 맞는 파일 찾기
    for ext in ALLOWED_EXTENSIONS:
        potential_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}{ext}")
        if os.path.exists(potential_path):  # 해당 파일이 존재하면
            file_path = potential_path
            output_path = os.path.join(PROCESSED_DIRECTORY, f"{file_id}{ext}")
            break

    if not file_path:
        raise HTTPException(status_code=404, detail="Image file not found")

    # 색감 보정 로직 처리 (여기서 adjust_white_balance 함수 적용)
    try:
        adjusted_path = adjust_white_balance(file_path, output_path)  # 실제 화이트 밸런스 보정 함수 호출
        media_type = get_media_type(adjusted_path)  # 확장자에 맞는 MIME 타입을 설정
        return FileResponse(adjusted_path, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/extract-color/")
async def extract_color():
    file_id = app.state.new_image_id
    # 고유 ID를 기준으로 이미지 파일 찾기
    file_path = None

    # 확장자에 맞는 파일 찾기
    for ext in ALLOWED_EXTENSIONS:
        potential_path = os.path.join(PROCESSED_DIRECTORY, f"{file_id}{ext}")
        if os.path.exists(potential_path):  # 해당 파일이 존재하면
            file_path = potential_path
            break

    if not file_path:
        raise HTTPException(status_code=404, detail="Image file not found")

    # 색깔 추출 로직 처리 (여기서 process_images 함수 적용)
    try:
        hex_colors = process_images(file_path)  # 실제 색상 추출 함수 호출
        personal_colors = get_personal_color(hex_colors)  # 색상 변환 함수 호출
        return JSONResponse(content=personal_colors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
