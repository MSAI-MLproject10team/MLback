import cv2
import numpy as np

# 화이트 밸런스 적용 함수
def adjust_white_balance(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    
    # 화이트 밸런스 조정 (Gray World Assumption)
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    balanced = image.copy()
    balanced[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    balanced[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
    
    # White Patch Algorithm 적용 (가장 밝은 부분을 기준으로 보정)
    max_b = np.max(balanced[:, :, 0])
    max_g = np.max(balanced[:, :, 1])
    max_r = np.max(balanced[:, :, 2])
    
    balanced[:, :, 0] = np.clip((balanced[:, :, 0] / max_b) * 255, 0, 255)
    balanced[:, :, 1] = np.clip((balanced[:, :, 1] / max_g) * 255, 0, 255)
    balanced[:, :, 2] = np.clip((balanced[:, :, 2] / max_r) * 255, 0, 255)
    
    # 감마 보정 적용 (더 밝게)
    gamma = 1.8  # 기존 1.2에서 증가
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(balanced.astype(np.uint8), gamma_table)
    
    # 채도 조정 (HSV 변환 후 S값 증가)
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # 채도 30% 증가
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 결과 저장
    # cv2.imwrite(image_path, cv2.cvtColor(saturated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, cv2.cvtColor(saturated, cv2.COLOR_RGB2BGR))
