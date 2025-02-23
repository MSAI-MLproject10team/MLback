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
    Convert hex color codes from image processing results to personal color types
    
    Args:
        hex_colors (Dict[str, str]): Dictionary with item types as keys and hex colors as values
            Example: {'pants': '#331519', 'top': '#e3cdd4'}
    
    Returns:
        Dict[str, str]: Dictionary with item types as keys and personal color types as values
            Example: {'pants': 'Dark Winter', 'top': 'Light Summer'}
    
    Raises:
        ValueError: If invalid hex color code is provided
    """
    results = {}
    
    for item_type, hex_color in hex_colors.items():
        # Ensure hex color starts with #
        if not hex_color.startswith('#'):
            hex_color = '#' + hex_color
            
        try:
            # Convert to HSV
            rgb = hex_to_rgb(hex_color)
            h, s, v = rgb_to_hsv(rgb)
            
            # Determine season based on hue, saturation, and value
            if 315 <= h or h < 45:  # Red to Yellow
                if v > 70 and s > 50:
                    season = "Spring"
                else:
                    season = "Autumn"
            elif 45 <= h < 165:  # Yellow-Green to Green
                if v > 70:
                    season = "Spring"
                else:
                    season = "Autumn"
            elif 165 <= h < 255:  # Blue-Green to Blue
                if v > 70:
                    season = "Summer"
                else:
                    season = "Winter"
            else:  # 255-315: Purple to Magenta
                if v > 70 and s < 50:
                    season = "Summer"
                else:
                    season = "Winter"
            
            # Detailed classification within each season
            if season == "Spring":
                if v > 85 and s < 60:
                    personal_color = "Light Spring"
                elif 60 <= s <= 85:
                    personal_color = "True Spring"
                else:
                    personal_color = "Bright Spring"
            
            elif season == "Summer":
                if v > 85 and s < 50:
                    personal_color = "Light Summer"
                elif 50 <= s <= 70:
                    personal_color = "True Summer"
                else:
                    personal_color = "Soft Summer"
            
            elif season == "Autumn":
                if s < 50 and v > 60:
                    personal_color = "Soft Autumn"
                elif 50 <= s <= 80:
                    personal_color = "True Autumn"
                else:
                    personal_color = "Dark Autumn"
            
            else:  # Winter
                if v < 50 and s > 50:
                    personal_color = "Dark Winter"
                elif 50 <= s <= 80:
                    personal_color = "True Winter"
                else:
                    personal_color = "Bright Winter"
                    
            results[item_type] = personal_color
            
        except ValueError as e:
            results[item_type] = f"Error: Invalid color code - {str(e)}"
            
    return results