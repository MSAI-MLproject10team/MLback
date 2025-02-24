from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uuid
import shutil

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

# 고유 ID를 생성하는 함수
def generate_unique_id():
    return str(uuid.uuid4())

def upload_new_image(file, UPLOAD_DIRECTORY):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # 확장자 검사
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=430, 
            detail=f"Invalid file type: {file_ext}. Only PNG, JPG, JPEG are allowed."
        )
    
    # 고유 ID 생성
    unique_id = generate_unique_id()
    
    # 고유 ID를 포함한 새로운 파일 경로 설정
    file_path = os.path.join(UPLOAD_DIRECTORY, f"{unique_id}{file_ext}")

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path, unique_id
