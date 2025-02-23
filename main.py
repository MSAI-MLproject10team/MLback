from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from fin2 import process_images
from colclass import ColorClassifierApp

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_images"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app.mount("/images", StaticFiles(directory=UPLOAD_DIRECTORY), name="images")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 1. 이미지 업로드
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. fin2.py를 사용하여 이미지 분석
        analysis_result = process_images(file_path)

        # 3. colclass.py를 사용하여 퍼스널 컬러 분류
        result = {}
        for category, (hex_code, probability) in analysis_result.items():
            color_classifier = ColorClassifierApp(None)
            personal_color = color_classifier.classify_personal_color(hex_code)
            rgb = color_classifier.hex_to_rgb(hex_code)
            hsv = color_classifier.rgb_to_hsv(rgb)

            result[category] = {
                "color": {
                    "hex_code": hex_code,
                    "rgb": rgb,
                    "hsv": {
                        "h": round(hsv[0], 1),
                        "s": round(hsv[1], 1),
                        "v": round(hsv[2], 1)
                    }
                },
                "probability": float(probability),
                "personal_color": personal_color
            }

        # 이미지 URL 추가
        image_url = f"/images/{file.filename}"
        result["image_url"] = image_url

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
