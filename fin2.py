import requests
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth




## Custom Vision 결과 확인
# 임계값 설정
THRESHOLD = 0.8  # 80% 이상 확률만 표시



def process_images(image_path):

    # 카테고리별 색상 저장용 딕셔너리
    category_colors = defaultdict(lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def detect_objects(image_path):
        # 이미지 로드
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # API 요청 헤더 및 데이터
        headers = {
            "Prediction-Key": prediction_key,
            "Content-Type": "application/octet-stream"
        }
        url = f"{prediction_endpoint}/customvision/v3.0/Prediction/{project_id}/detect/iterations/{model_name}/image"
        
        # 요청 전송
        response = requests.post(url, headers=headers, data=image_data)
        
        if response.status_code != 200:
            print("Error:", response.text)
            return None

        # JSON 응답 데이터 파싱
        return response.json()

    def visualize_detections(image_path, detections):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for prediction in detections["predictions"]:
            probability = prediction["probability"]
            if probability < THRESHOLD:
                continue  # 임계값보다 낮으면 무시

            tag_name = prediction["tagName"]
            bbox = prediction["boundingBox"]

            # 바운딩 박스 정보 변환
            h, w, _ = image.shape
            x1, y1 = int(bbox["left"] * w), int(bbox["top"] * h)
            x2, y2 = int((bbox["left"] + bbox["width"]) * w), int((bbox["top"] + bbox["height"]) * h)

            # 카테고리별 색상 선택
            color = category_colors[tag_name]

            # 바운딩 박스 및 텍스트 추가
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{tag_name} ({probability:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # # 결과 시각화
        # plt.figure(figsize=(10, 8))
        # plt.imshow(image)
        # plt.axis("off")
        # plt.show()

    # 실행 예시

    detections = detect_objects(image_path)
    if detections:
        visualize_detections(image_path, detections)


    ## 필요한 부분만 가져오기
    def show_cropped_objects_clean(image_path, detections, threshold=THRESHOLD):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cropped_images = []  # 크롭된 이미지를 저장할 리스트

        for prediction in detections["predictions"]:
            probability = prediction["probability"]
            if probability < threshold:
                continue  # 임계값보다 낮으면 무시

            tag_name = prediction["tagName"]
            bbox = prediction["boundingBox"]

            # 바운딩 박스 좌표 변환
            h, w, _ = image.shape
            x1, y1 = int(bbox["left"] * w), int(bbox["top"] * h)
            x2, y2 = int((bbox["left"] + bbox["width"]) * w), int((bbox["top"] + bbox["height"]) * h)

            # 객체 부분 잘라내기
            cropped_object = image[y1:y2, x1:x2]
            cropped_images.append((tag_name, probability, cropped_object))  # (라벨, 확률, 이미지) 저장

            # # 새 창에 개별 이미지 출력
            # plt.figure(figsize=(3, 3))
            # plt.imshow(cropped_object)
            # plt.title(f"{tag_name} ({probability:.2f})")
            # plt.axis("off")
            # plt.show()

        return cropped_images  # 크롭된 이미지 리스트 반환

    # 실행 예시
    cropped_objects = show_cropped_objects_clean(image_path, detections)

    # 반환된 이미지 확인
    # for tag, prob, img in cropped_objects:
    #     print(f"Tag: {tag}, Probability: {prob:.2f}, Image Shape: {img.shape}")





    def remove_background(cropped_images):
        final_images = []  # 최종 결과 저장

        for tag, prob, img in cropped_images:
            h, w, _ = img.shape
            
            # 초기 마스크 설정
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # GrabCut을 위한 초기 사각형 (조금 작게 설정)
            rect = (5, 5, w-10, h-10)
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 배경과 전경 분리
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            result = img * mask2[:, :, np.newaxis]
            
            # 경계 부드럽게 (가우시안 블러 적용)
            blurred = cv2.GaussianBlur(result, (5, 5), 0)

            # 알파 채널 추가
            alpha = mask2 * 255
            rgba = np.dstack([blurred, alpha])

            # # 📌 최종 이미지 출력
            # plt.figure(figsize=(4, 4))
            # plt.imshow(rgba)
            # plt.title(f"{tag} (Transparent Background)", fontsize=12)
            # plt.axis("off")
            # plt.show()

            final_images.append((tag, rgba))  # 결과 저장
        
        return final_images  # 최종 이미지 반환

    # 🔥 실행
    final_results = remove_background(cropped_objects)


    def get_most_dominant_color(final_results, quantile=0.2, n_samples=1000):
        """
        각 이미지에서 가장 비율이 높은 색상의 hex code만 반환하는 함수
        
        Parameters:
        -----------
        final_results : list of tuples
            (tag, rgba_img) 형태의 튜플을 담은 리스트
        quantile : float
            Mean-Shift 클러스터링의 bandwidth 추정에 사용할 quantile 값
        n_samples : int
            bandwidth 추정에 사용할 샘플 수
        
        Returns:
        --------
        dict
            이미지 태그를 키로 하고, (hex_code, percentage) 튜플을 값으로 하는 딕셔너리
        """
        dominant_colors = {}
        
        for tag, rgba_img in final_results:
            mask = rgba_img[:, :, 3] > 0
            pixels = rgba_img[mask][:, :3]
            
            if len(pixels) == 0:
                dominant_colors[tag] = None
                continue
                
            bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=n_samples)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            
            labels = ms.fit_predict(pixels)
            cluster_centers = ms.cluster_centers_.astype(int)
            
            counts = np.bincount(labels)
            dominant_label = np.argmax(counts)
            dominant_color = cluster_centers[dominant_label]
            percentage = (counts[dominant_label] / len(pixels)) * 100
            
            hex_code = '#{:02x}{:02x}{:02x}'.format(
                dominant_color[0],
                dominant_color[1],
                dominant_color[2]
            )
            
            #dominant_colors[tag] = (hex_code, percentage)
            dominant_colors[tag] = hex_code
        
        return dominant_colors

    # for tag, prob, img in cropped_objects:
    #     print(f"Tag: {tag}, Probability: {prob:.2f}, Image Shape: {img.shape}")


    # 실행 방법
    results = get_most_dominant_color(final_results)
    # for tag, color_info in results.items():
    #     if color_info:
    #         hex_code, percentage = color_info
    #         print(f"{tag}: {hex_code} ({percentage:.1f}%)")
    #     else:
    #         print(f"{tag}: No valid pixels found")
        

    return results

#이미지 경로
image_file = f"data/musinsa_images_pants_ver2/pants_313.jpg"
show= process_images(image_file)
print(show)
print(type(show))

