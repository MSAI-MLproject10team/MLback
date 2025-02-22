from fastapi import FastAPI, File, UploadFile
import shutil
import uvicorn
#from finalmodel import detect_objects, visualize_detections, show_cropped_objects_clean, remove_background, get_most_dominant_color
from fin2 import process_images
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message":"Hello, World"}


# 이미지 업로드 엔드포인트
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Invalid file type. Only JPEG and PNG are allowed."}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "image_path": file_path}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)