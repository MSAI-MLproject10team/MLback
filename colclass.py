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
    return h * 360, s * 100, v * 100  # Convert to degrees (H), percentage (S, V)

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

            # Spring colors (20° ~ 80°)
            if 20 <= h <= 80:
                if v >= 75 and s >= 60:
                    personal_color = "Bright Spring"
                elif v >= 70 and s < 50:
                    personal_color = "Light Spring"
                else:
                    personal_color = "True Spring"

            # Summer colors (180° ~ 260°)
            elif 180 <= h <= 260:
                if v >= 70 and 30 <= s <= 50:
                    personal_color = "Light Summer"
                elif v >= 50:
                    personal_color = "True Summer"
                else:
                    personal_color = "Soft Summer"

            # Autumn colors (80° ~ 180°)
            elif 80 <= h <= 180:
                if v <= 50 and s >= 50:
                    personal_color = "Deep Autumn"
                elif v <= 65 and s >= 40:
                    personal_color = "True Autumn"
                else:
                    personal_color = "Soft Autumn"

            # Winter colors (260° ~ 360°)
            else:
                if v <= 50 and s >= 40:
                    personal_color = "Dark Winter"
                elif v >= 70 and s >= 60:
                    personal_color = "Bright Winter"
                else:
                    personal_color = "True Winter"

            results[item_type] = personal_color

        except ValueError as e:
            results[item_type] = f"Error: Invalid color code - {str(e)}"

    return results
