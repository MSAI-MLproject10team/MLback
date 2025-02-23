import colorsys
from typing import Dict, Tuple

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert HEX color code to RGB values"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB values to HSV values"""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return h * 360, s * 100, v * 100

def get_personal_color(hex_colors: Dict[str, str]) -> Dict[str, str]:
    """
    Convert hex color codes to personal color types using standard HSV ranges
    
    Args:
        hex_colors (Dict[str, str]): Dictionary with item types as keys and hex colors as values
    
    Returns:
        Dict[str, str]: Dictionary with item types as keys and personal color types as values
    """
    results = {}
    
    for item_type, hex_color in hex_colors.items():
        if not hex_color.startswith('#'):
            hex_color = '#' + hex_color
            
        try:
            rgb = hex_to_rgb(hex_color)
            h, s, v = rgb_to_hsv(rgb)
            
            # Personal color classification based on HSV ranges
            if 20 <= h <= 60:  # 봄/가을 계열
                if s >= 40 and v >= 70:  # 봄 라이트
                    personal_color = "Light Spring"
                elif s >= 50 and 60 <= v <= 85:  # 봄 트루
                    personal_color = "True Spring"
                elif s >= 60 and v >= 75:  # 봄 브라이트
                    personal_color = "Bright Spring"
                elif 10 <= s <= 40 and 30 <= v <= 60:  # 가을 소프트
                    personal_color = "Soft Autumn"
                elif 40 <= s <= 70 and 30 <= v <= 65:  # 가을 트루
                    personal_color = "True Autumn"
                elif s >= 50 and 20 <= v <= 50:  # 가을 딥
                    personal_color = "Deep Autumn"
                else:
                    personal_color = "Neutral"
                    
            elif 180 <= h <= 260:  # 여름 계열
                if 30 <= s <= 50 and v >= 70:  # 여름 라이트
                    personal_color = "Light Summer"
                elif 20 <= s <= 50 and 50 <= v <= 80:  # 여름 트루
                    personal_color = "True Summer"
                elif 10 <= s <= 40 and 40 <= v <= 70:  # 여름 소프트
                    personal_color = "Soft Summer"
                else:
                    personal_color = "Neutral"
                    
            elif 260 <= h <= 350:  # 겨울 계열
                if s >= 40 and 20 <= v <= 50:  # 겨울 다크
                    personal_color = "Dark Winter"
                elif s >= 50 and 50 <= v <= 80:  # 겨울 트루
                    personal_color = "True Winter"
                elif s >= 60 and v >= 70:  # 겨울 브라이트
                    personal_color = "Bright Winter"
                else:
                    personal_color = "Neutral"
                    
            else:
                personal_color = "Neutral"
                
            results[item_type] = personal_color
            
        except ValueError as e:
            results[item_type] = f"Error: Invalid color code - {str(e)}"
            
    return results