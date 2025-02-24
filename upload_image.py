from fastapi import FastAPI, File, UploadFile, HTTPException
import os

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # 허용할 이미지 확장자

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
    
    # 1. 이미지 업로드
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

    return file_path, file.filename
