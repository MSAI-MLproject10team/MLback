import tkinter as tk
from typing import Tuple
import colorsys

class ColorClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Color Classifier")
        
        # 입력 프레임
        input_frame = tk.Frame(root, pady=10)
        input_frame.pack()
        
        tk.Label(input_frame, text="Enter HEX color (e.g., #234567):").pack(side=tk.LEFT)
        self.color_entry = tk.Entry(input_frame)
        self.color_entry.pack(side=tk.LEFT, padx=5)
        self.color_entry.insert(0, "#234567")
        
        tk.Button(input_frame, text="Classify", command=self.update_color).pack(side=tk.LEFT)
        
        # 결과 프레임
        result_frame = tk.Frame(root, pady=10)
        result_frame.pack()
        
        # 색상 표시 캔버스
        self.canvas = tk.Canvas(result_frame, width=200, height=200)
        self.canvas.pack()
        
        # 결과 텍스트
        self.result_label = tk.Label(result_frame, text="", pady=10)
        self.result_label.pack()
        
        # 초기 색상 표시
        self.update_color()
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """HEX 컬러 코드를 RGB 값으로 변환"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """RGB 값을 HSV 값으로 변환"""
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        return h * 360, s * 100, v * 100

    def classify_personal_color(self, hex_color: str) -> str:
        """HEX 컬러 코드를 퍼스널 컬러로 분류"""
        rgb = self.hex_to_rgb(hex_color)
        h, s, v = self.rgb_to_hsv(rgb)
        
        # 계절 결정
        if 315 <= h or h < 45:  # 빨강~노랑
            if v > 70 and s > 50:
                season = "Spring"
            else:
                season = "Autumn"
        elif 45 <= h < 165:  # 노랑초록~초록
            if v > 70:
                season = "Spring"
            else:
                season = "Autumn"
        elif 165 <= h < 255:  # 청록~파랑
            if v > 70:
                season = "Summer"
            else:
                season = "Winter"
        else:  # 255~315: 보라~자주
            if v > 70 and s < 50:
                season = "Summer"
            else:
                season = "Winter"
        
        # 세부 분류
        if season == "Spring":
            if v > 85 and s < 60:
                return "Light Spring"
            elif 60 <= s <= 85:
                return "True Spring"
            else:
                return "Bright Spring"
        
        elif season == "Summer":
            if v > 85 and s < 50:
                return "Light Summer"
            elif 50 <= s <= 70:
                return "True Summer"
            else:
                return "Soft Summer"
        
        elif season == "Autumn":
            if s < 50 and v > 60:
                return "Soft Autumn"
            elif 50 <= s <= 80:
                return "True Autumn"
            else:
                return "Dark Autumn"
        
        else:  # Winter
            if v < 50 and s > 50:
                return "Dark Winter"
            elif 50 <= s <= 80:
                return "True Winter"
            else:
                return "Bright Winter"

    def update_color(self):
        """색상 업데이트 및 결과 표시"""
        try:
            hex_color = self.color_entry.get()
            if not hex_color.startswith('#'):
                hex_color = '#' + hex_color
            
            # 색상 표시
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, 200, 200, fill=hex_color)
            
            # 퍼스널 컬러 분류
            personal_color = self.classify_personal_color(hex_color)
            
            # 결과 표시
            rgb = self.hex_to_rgb(hex_color)
            hsv = self.rgb_to_hsv(rgb)
            result_text = f"HEX: {hex_color}\n"
            result_text += f"RGB: {rgb}\n"
            result_text += f"HSV: {hsv[0]:.1f}°, {hsv[1]:.1f}%, {hsv[2]:.1f}%\n"
            result_text += f"Personal Color Type: {personal_color}"
            
            self.result_label.config(text=result_text)
            
        except ValueError:
            self.result_label.config(text="Invalid color code")

def main():
    root = tk.Tk()
    app = ColorClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()