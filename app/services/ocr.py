import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Dict, Any, Tuple, List, Union
from dataclasses import dataclass
import os

@dataclass
class RegionConfig:
    """Configuration for a specific region in the dialog box."""
    x: int  # X coordinate from left
    y: int  # Y coordinate from top
    w: int  # Width
    h: int  # Height
    padding: int  # Padding around the region
    ocr_config: str  # Tesseract configuration for this region

class DialogConfig:
    """Configuration for all regions in the dialog box."""
    # Dialog box dimensions
    DIALOG_X = 129  # X coordinate of dialog box
    DIALOG_Y = 569  # Y coordinate of dialog box
    DIALOG_W = 822  # Width of dialog box
    DIALOG_H = 1136  # Height of dialog box
    
    # Region coordinates are relative to dialog box
    NAME_REGION = RegionConfig(
        x=50,    # Name starts near the top
        y=20,    # Small offset from top
        w=780,   # Most of the width
        h=80,    # Enough height for name
        padding=5,
        ocr_config='--oem 3 --psm 7'  # Single line
    )
    
    TYPE_REGION = RegionConfig(
        x=220,    # Left aligned
        y=140,   # Below name
        w=400,   # Half width for type
        h=50,    # Height for type text
        padding=5,
        ocr_config='--oem 3 --psm 6'  # Uniform block
    )
    
    QUALITY_REGION = RegionConfig(
        x=220,   # Right half
        y=200,   # Same height as type
        w=400,   # Half width for quality
        h=50,    # Same height as type
        padding=5,
        ocr_config='--oem 3 --psm 6'  # Uniform block
    )
    
    STATS_REGION = RegionConfig(
        x=200,    # Left aligned
        y=130,   # Below type/quality
        w=450,   # Most of the width
        h=200,   # Tall enough for all stats
        padding=5,
        ocr_config='--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789HPATKDEFCRITSPDhpatkdefcritspd+-%., "'
    )
    
    EFFECTS_REGION = RegionConfig(
        x=50,    # Left aligned
        y=350,   # Below stats
        w=700,   # Most of the width
        h=800,   # Remaining height for effects
        padding=5,
        ocr_config='--oem 3 --psm 6'  # Uniform block
    )

class OCRService:
    @staticmethod
    def crop_region(image: np.ndarray, config: RegionConfig) -> np.ndarray:
        """Crop a region from image using config coordinates."""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Adjust coordinates to fit within image
        x1 = max(0, min(config.x - config.padding, width))
        y1 = max(0, min(config.y - config.padding, height))
        x2 = max(0, min(config.x + config.w + config.padding, width))
        y2 = max(0, min(config.y + config.h + config.padding, height))
        
        # Check if crop region is valid
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)  # Return tiny blank image
        
        # Save cropped region for debugging
        cropped = image[y1:y2, x1:x2]
        debug_dir = "debug_regions"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Generate a unique filename based on coordinates
        filename = f"region_x{x1}y{y1}w{x2-x1}h{y2-y1}.jpg"
        debug_path = os.path.join(debug_dir, filename)
        cv2.imwrite(debug_path, cropped)
            
        return cropped

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Resize image to make text clearer
        scale_factor = 2
        resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrasted = clahe.apply(denoised)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Save preprocessed image for debugging
        debug_dir = "debug_regions"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        # Generate a unique filename
        filename = f"preprocessed_{np.random.randint(10000)}.jpg"
        debug_path = os.path.join(debug_dir, filename)
        cv2.imwrite(debug_path, cleaned)
        
        return cleaned

    @staticmethod
    def process_name_region(image: np.ndarray, config: RegionConfig) -> str:
        """Process equipment name region."""
        # Crop and preprocess
        region = OCRService.crop_region(image, config)
        processed = OCRService.preprocess_for_ocr(region)
        
        # OCR with specific config
        text = pytesseract.image_to_string(processed, config=config.ocr_config).strip()
        return text

    @staticmethod
    def process_type_and_quality(image: np.ndarray, type_config: RegionConfig, quality_config: RegionConfig) -> Tuple[str, str]:
        """Process type and quality regions."""
        # Process type
        type_region = OCRService.crop_region(image, type_config)
        type_processed = OCRService.preprocess_for_ocr(type_region)
        type_text = pytesseract.image_to_string(type_processed, config=type_config.ocr_config)
        
        # Process quality
        quality_region = OCRService.crop_region(image, quality_config)
        quality_processed = OCRService.preprocess_for_ocr(quality_region)
        quality_text = pytesseract.image_to_string(quality_processed, config=quality_config.ocr_config)
        
        # Extract type and quality
        type_match = re.search(r'(Weapon|Armor|Ring|Necklace|Shoe|Boot|Shield|Glove|Belt|Hat)\s*Type', type_text, re.IGNORECASE)
        quality_match = re.search(r'Quality\s+(Common|Uncommon|Rare|Epic|Legendary|Mythic|Advanced|Ultimate)', quality_text, re.IGNORECASE)
        
        equip_type = type_match.group(1) if type_match else ''
        quality = quality_match.group(1) if quality_match else ''
        
        return equip_type, quality

    @staticmethod
    def process_stats_region(image: np.ndarray, config: RegionConfig) -> Dict[str, Union[int, float]]:
        """Process equipment stats region."""
        # Crop and preprocess
        region = OCRService.crop_region(image, config)
        processed = OCRService.preprocess_for_ocr(region)
        
        # OCR with specific config
        text = pytesseract.image_to_string(processed, config=config.ocr_config)
        
        stats = {
            'hp': 0, 'atk': 0, 'def': 0,
            'crit': 0.0, 'atk_spd': 0.0, 'evasion': 0.0
        }
        
        # Extract stats
        for line in text.split('\n'):
            hp_match = re.search(r'HP\s*[+-]?(\d[\d,]*)', line, re.IGNORECASE)
            if hp_match:
                stats['hp'] = int(hp_match.group(1).replace(',', ''))
                
            atk_match = re.search(r'ATK\s*[+-]?(\d[\d,]*)', line, re.IGNORECASE)
            if atk_match:
                stats['atk'] = int(atk_match.group(1).replace(',', ''))
                
            def_match = re.search(r'DEF\s*[+-]?(\d[\d,]*)', line, re.IGNORECASE)
            if def_match:
                stats['def'] = int(def_match.group(1).replace(',', ''))
                
            crit_match = re.search(r'CRIT\s*[+-]?(\d+\.?\d*)', line, re.IGNORECASE)
            if crit_match:
                stats['crit'] = float(crit_match.group(1))
                
            spd_match = re.search(r'SPD\s*[+-]?(\d+\.?\d*)', line, re.IGNORECASE)
            if spd_match:
                stats['atk_spd'] = float(spd_match.group(1))
        
        return stats

    @staticmethod
    def auto_correct_effect_str(effect_str: str) -> str:
        regex_corrections = [
            (r'b[%o0]ss', 'boss'),
            (r'p[%o0]imate', 'primate'),
            (r'a[%o0]imal', 'animal'),
            (r'u[%o0]ndead', 'undead'),
            (r'd[%o0]mon', 'demon'),
            (r'c[%o0]itical', 'critical'),
            (r'h[%o0]t', 'hit'),
            (r's[%o0]oe', 'shoe'),
            (r'g[%o0]ove', 'glove'),
            (r'r[%o0]ng', 'ring'),
            (r'n[%o0]cklace', 'necklace'),
            (r'b[%o0]lt', 'belt'),
            (r'h[%o0]lm', 'helm'),
            (r'a[%o0]mor', 'armor'),
            (r'evasi[%o0]n', 'Evasion'),
            (r'sl[%o0]t', 'slot'),
            (r'weap[%o0]n', 'Weapon'),
            (r'w[%o0]re', 'wore'),
            (r'[%o0]ld', 'old'),
            (r'c%nsumpti%n', 'consumption'),
            (r'satiety', 'satiety'),
            (r'Exp', 'exp'),
            (r'w[%o0]rn', 'worn'),
            (r'[%o0]riginal', 'original'),
            (r'agains:', 'against'),
            (r'b[%o0]{2}ts', 'Boots'),
            (r'fr[%o0]m', 'from'),
            (r'[%o0]f', 'of'),
            (r'chance [%o0]f', 'chance of'),
            (r'decreas[%o0]ng', 'reducing'),
            (r'decreasing', 'reducing'),
            (r'damage taken', 'damage_received'),
            (r'suphemel', 'Supreme'),
            (r'atk5pd', 'ATK SPD'),
            (r'criticai', 'Critical'),
            (r'criticaI', 'Critical'),
            (r'lifesteai', 'Lifesteal'),
            (r'lifesteaI', 'Lifesteal'),
            (r'm[%o0]{2}d', 'Mood'),
            (r'm[%o0]vement', 'Movement'),
            (r'&', ''),
            (r'["""]', ''),
            (r'¢\+', ''),
            (r'¢', ''),
            (r'~', ''),
            (r'©', ''),
            (r'@', ''),
            (r'»', ''),
            (r'%f', 'of'),
            (r'casting', 'cast'),
            (r'bh:', 'by'),
            (r'extra materials', 'bonus_materials'),
            (r'getting', 'gain'),
            (r'Critical Hit damage', 'crit_damage'),
            (r'Critical Hit chance', 'crit_chance'),
            (r'ATK SPD', 'atk_spd'),
            (r'ATK', 'atk'),
            (r'DEF', 'def'),
            (r'HP', 'hp'),
            (r'Evasion', 'evasion'),
            (r'Movement Speed', 'move_speed'),
            (r'(\d)\s+(\d)%', r'\1\2%'),  # nối số bị split như '1 6%' -> '16%'
            (r'(\d)[^\d%]+%', r'\1%'),     # loại ký tự lạ giữa số và %
        ]
        for pattern, repl in regex_corrections:
            effect_str = re.sub(pattern, repl, effect_str, flags=re.IGNORECASE)
        effect_str = re.sub(r'^[^A-Za-z0-9]+', '', effect_str)
        return effect_str

    @staticmethod
    def normalize_effect_str(effect_str: str) -> str:
        effect_str = effect_str.strip()
        effect_str = re.sub(r'^[^A-Za-z0-9]+', '', effect_str)
        effect_str = re.sub(r'\s+', ' ', effect_str)
        effect_str = re.sub(r'[.,;:!?]+$', '', effect_str)
        return effect_str

    @staticmethod
    def parse_effect(effect_str: str) -> Tuple[Dict[str, float], str]:
        original_str = effect_str
        effect_str = OCRService.auto_correct_effect_str(effect_str)
        effect_str = OCRService.normalize_effect_str(effect_str)
        effect_str = effect_str.replace('°', '%').replace('º', '%').replace('o', '%')
        if 'empty rune slot' in effect_str.lower():
            return ({}, None)
        # Parse compound effects: 'Increase atk by [VALUE1]% and def by [VALUE2]%' or similar
        m = re.match(r"Increase ([a-z_]+) by (\d+)% and ([a-z_]+) by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            stat1 = m.group(1).strip().lower()
            value1 = float(m.group(2))
            stat2 = m.group(3).strip().lower()
            value2 = float(m.group(4))
            stat_map = {
                'atk': 'attack',
                'def': 'defense',
                'hp': 'hp',
                'atk_spd': 'attack_speed',
                'crit_damage': 'critical_hit_damage',
                'crit_chance': 'critical_hit_chance',
                'evasion': 'evasion',
                'move_speed': 'movement_speed',
            }
            stat1_key = stat_map.get(stat1, stat1)
            stat2_key = stat_map.get(stat2, stat2)
            return ({f"{stat1_key}_percent": value1, f"{stat2_key}_percent": value2}, None)
        # Nếu có nhiều số liệu trong một dòng, tự động split và parse từng phần
        if re.search(r'\d+%.*\d+%', effect_str):
            parts = re.split(r'(?<=\d%)\s*(?:and|,|\+|\s)\s*', effect_str)
            parsed = {}
            unparsed = []
            for part in parts:
                d, un = OCRService.parse_effect(part)
                parsed.update(d)
                if un:
                    unparsed.append(un)
            if parsed:
                return (parsed, None if not unparsed else ', '.join(unparsed))
        if '+' in effect_str:
            parts = [p.strip() for p in effect_str.split('+') if p.strip()]
            parsed = {}
            unparsed = []
            for part in parts:
                d, un = OCRService.parse_effect(part)
                parsed.update(d)
                if un:
                    unparsed.append(un)
            if parsed:
                return (parsed, None if not unparsed else ', '.join(unparsed))
        if ',' in effect_str:
            parts = [p.strip() for p in effect_str.split(',') if p.strip()]
            parsed = {}
            unparsed = []
            for part in parts:
                d, un = OCRService.parse_effect(part)
                parsed.update(d)
                if un:
                    unparsed.append(un)
            if parsed:
                return (parsed, None if not unparsed else ', '.join(unparsed))
        # 1. Increase atk_spd by [VALUE]%
        m = re.match(r"Increase atk_spd by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"atk_spd_percent": value}, None)
        # 2. Increase crit_damage by [VALUE]%
        m = re.match(r"Increase crit_damage by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"critical_hit_damage_percent": value}, None)
        # 3. Increase crit_chance by [VALUE]%
        m = re.match(r"Increase crit_chance by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"critical_hit_chance_percent": value}, None)
        # 4. Increase def by [VALUE]%
        m = re.match(r"Increase def by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"def_percent": value}, None)
        # 5. Increase hp by [VALUE]%
        m = re.match(r"Increase hp by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"hp_percent": value}, None)
        # 6. Increase evasion by [VALUE]%
        m = re.match(r"Increase evasion by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"evasion_percent": value}, None)
        # 7. Increase move_speed by [VALUE]%
        m = re.match(r"Increase move_speed by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"movement_speed_percent": value}, None)
        # 8. Increase atk by [VALUE]%
        m = re.match(r"Increase atk by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"atk_percent": value}, None)
        # 9. Increase Against [Target] type [VALUE]% damage (target thường)
        m = re.match(r"Increase Against ([a-z_]+) type (\d+)% damage", effect_str, re.IGNORECASE)
        if m:
            target = m.group(1).strip().lower()
            value = float(m.group(2))
            return ({f"damage_vs_{target}_percent": value}, None)
        # 1. Increase [STAT] by [VALUE]%
        m = re.match(r"Increase ([A-Za-z ]+) by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            stat = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"{stat}_percent": value}, None)
        # 1b. Increase [STAT] [VALUE]% (không có 'by')
        m = re.match(r"Increase ([A-Za-z ]+) (\d+)%", effect_str, re.IGNORECASE)
        if m:
            stat = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"{stat}_percent": value}, None)
        # 1c. Increase [STAT] chance by [VALUE]%
        m = re.match(r"Increase ([A-Za-z ]+) chance by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            stat = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"{stat}_chance_percent": value}, None)
        # 2. Increase [VALUE]% damage against [TARGET] types?
        m = re.match(r"Increase (\d+)% damage against ([A-Za-z ]+?)(?: types?)?", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            target = m.group(2).strip().lower().replace(' ', '_')
            return ({f"damage_vs_{target}_percent": value}, None)
        # 2b. Increase Against [Target] type [VALUE]% damage
        m = re.match(r"Increase Against ([A-Za-z ]+) type (\d+)% damage", effect_str, re.IGNORECASE)
        if m:
            target = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"damage_vs_{target}_percent": value}, None)
        # 2c. Decrease [VALUE]% damage against [TARGET] types?
        m = re.match(r"Decrease (\d+)% damage against ([A-Za-z ]+?)(?: types?)?", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            target = m.group(2).strip().lower().replace(' ', '_')
            return ({f"damage_vs_{target}_percent": -value}, None)
        # 2d. Increase Movement Speed by [VALUE]%
        m = re.match(r"Increase Movement Speed by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"movement_speed_percent": value}, None)
        # 2e. Increase Evasion by [VALUE]%
        m = re.match(r"Increase Evasion by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"evasion_percent": value}, None)
        # 2f. Decrease Against [Target] type [VALUE]% damage
        m = re.match(r"Decrease Against ([A-Za-z ]+) type (\d+)% damage", effect_str, re.IGNORECASE)
        if m:
            target = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"damage_vs_{target}_percent": -value}, None)
        # 2g. Increase Against [Target] type [VALUE]% damage
        m = re.match(r"Increase Against ([A-Za-z ]+) type (\d+)% damage", effect_str, re.IGNORECASE)
        if m:
            target = m.group(1).strip().lower().replace(' ', '_')
            value = float(m.group(2))
            return ({f"damage_vs_{target}_percent": value}, None)
        # 3. [VALUE]% chance of casting
        m = re.match(r"(\d+)% chance of cast", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"chance_of_cast_percent": value}, None)
        # 4. [VALUE]% chance to cast
        m = re.match(r"(\d+)% chance to cast", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"chance_to_cast_percent": value}, None)
        # 5. by [VALUE]% (standalone)
        m = re.match(r"by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"by_percent": value}, None)
        # 6. Increase [VALUE]% EXP gain
        m = re.match(r"Increase (\d+)% Exp gain", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"exp_gain_percent": value}, None)
        # 7. [VALUE]% chance of [verb] [target]
        m = re.match(r"(\d+)% chance of ([a-z]+) ([a-z_]+)", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            verb = m.group(2).strip().lower()
            target = m.group(3).strip().lower()
            return ({f"chance_{verb}_{target}_percent": value}, None)
        # 8. Decrease [VALUE]% Mood consumption
        m = re.match(r"Decrease (\d+)% Mood consumption", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"mood_consumption_percent": -value}, None)
        # 9. Lifesteal [VALUE]% of total damage
        m = re.match(r"Lifesteal (\d+)% of total damage", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"lifesteal_percent": value}, None)
        # 10. [VALUE]% chance to transform into a [TARGET]
        m = re.match(r"(\d+)% chance to transform into a ([A-Za-z ]+)", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            target = m.group(2).strip().lower().replace(' ', '_')
            return ({f"chance_transform_into_{target}_percent": value}, None)
        # 11. [VALUE]% chance to [verb] [target]
        m = re.match(r"(\d+)% chance to ([a-z]+) ([a-z_]+)", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            verb = m.group(2).strip().lower()
            target = m.group(3).strip().lower()
            return ({f"chance_to_{verb}_{target}_percent": value}, None)
        # 12. Decrease [VALUE]% Mood consumption
        m = re.match(r"Decrease (\d+)% Mood consumption", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"mood_consumption_percent": -value}, None)
        # 13. Lifesteal [VALUE]% of total damage
        m = re.match(r"Lifesteal (\d+)% of total damage", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"lifesteal_percent": value}, None)
        # 14. [VALUE]% chance to transform into a [TARGET]
        m = re.match(r"(\d+)% chance to transform into a ([A-Za-z ]+)", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            target = m.group(2).strip().lower().replace(' ', '_')
            return ({f"chance_transform_into_{target}_percent": value}, None)
        # 15. [VALUE]% chance to [verb] [target]
        m = re.match(r"(\d+)% chance to ([a-z]+) ([a-z_]+)", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            verb = m.group(2).strip().lower()
            target = m.group(3).strip().lower()
            return ({f"chance_to_{verb}_{target}_percent": value}, None)
        # 1. [VALUE]% chance of reducing damage_received
        m = re.match(r"(\d+)% chance of reducing damage_received", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"chance_reduce_damage_received_percent": value}, None)
        # 2. [VALUE]% chance to reduce damage_received
        m = re.match(r"(\d+)% chance to reduce damage_received", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"chance_reduce_damage_received_percent": value}, None)
        # 3. [VALUE]% damage_received
        m = re.match(r"(\d+)% damage_received", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"damage_received_percent": value}, None)
        # 1. [VALUE]% chance of gain bonus_materials
        m = re.match(r"(\d+)% chance of gain bonus_materials", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"chance_gain_bonus_materials_percent": value}, None)
        # 2. Decrease crit_damage by [VALUE]%
        m = re.match(r"Decrease crit_damage by (\d+)%", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"crit_damage_percent": -value}, None)
        # 10. Increase Against boss type [VALUE]% damage
        m = re.match(r"Increase Against boss type (\d+)% damage", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"damage_vs_boss_percent": value}, None)
        # Decrease [VALUE]% exp gain
        m = re.match(r"Decrease (\d+)% exp gain", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"exp_gain_percent": -value}, None)
        # Increase [VALUE]% satiety consumption
        m = re.match(r"Increase (\d+)% satiety consumption", effect_str, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            return ({"satiety_consumption_percent": value}, None)
        return ({}, f"original: {original_str} | corrected: {effect_str}")

    @staticmethod
    def process_effects_region(image: np.ndarray, config: RegionConfig) -> Dict[str, Any]:
        """Process additional effects region and return both raw and structured effects."""
        # Crop and preprocess
        region = OCRService.crop_region(image, config)
        processed = OCRService.preprocess_for_ocr(region)
        # OCR with specific config
        text = pytesseract.image_to_string(processed, config=config.ocr_config)
        effects = []
        current_effect = ""
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if any(effect_start in line for effect_start in ['Increase', 'Decrease', 'Lifesteal', 'Empty rune']):
                if current_effect:
                    effects.append(current_effect)
                current_effect = line
            elif current_effect:
                current_effect += " " + line
                if any(end in current_effect.lower() for end in ['damage', 'consumption', 'slot', '%']):
                    effects.append(current_effect)
                    current_effect = ""
        if current_effect:
            effects.append(current_effect)
        effects = [
            effect for effect in effects
            if len(effect) > 5 and not any(skip in effect.lower() for skip in ['number of attempts', 'the old', 'warm new'])
        ]
        # Parse effects
        parsed = {}
        unparsed = []
        for eff in effects:
            d, un = OCRService.parse_effect(eff)
            parsed.update(d)
            if un:
                unparsed.append(un)
        return {
            'effects': effects,
            'parsed_effects': parsed,
            'unparsed_effects': unparsed
        }

    @staticmethod
    def save_debug_regions(image: np.ndarray, base_path: str):
        """Save debug images for each region."""
        # Create a copy for visualization
        debug_img = image.copy()
        
        # Draw and save each region
        regions = {
            'name': DialogConfig.NAME_REGION,
            'type': DialogConfig.TYPE_REGION,
            'quality': DialogConfig.QUALITY_REGION,
            'stats': DialogConfig.STATS_REGION,
            'effects': DialogConfig.EFFECTS_REGION
        }
        
        colors = {
            'name': (255, 0, 0),      # Blue
            'type': (0, 255, 0),      # Green
            'quality': (0, 0, 255),   # Red
            'stats': (255, 255, 0),   # Cyan
            'effects': (255, 0, 255)  # Magenta
        }
        
        for name, config in regions.items():
            # Draw rectangle on debug image
            cv2.rectangle(
                debug_img,
                (config.x, config.y),
                (config.x + config.w, config.y + config.h),
                colors[name],
                2
            )
            
            # Save individual region
            region = OCRService.crop_region(image, config)
            region_path = f"{base_path}_{name}.jpg"
            cv2.imwrite(region_path, region)
        
        # Save debug image with all regions marked
        debug_path = f"{base_path}_regions.jpg"
        cv2.imwrite(debug_path, debug_img)

    @staticmethod
    def extract_stats(image_path: str) -> Dict[str, Any]:
        """Extract equipment stats from image using fixed coordinates."""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'name': '',
                'type': '',
                'rarity': '',
                'stats': {
                    'hp': 0, 'atk': 0, 'def': 0,
                    'crit': 0.0, 'atk_spd': 0.0, 'evasion': 0.0
                },
                'additional_effects': {'effects': []}
            }
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Adjust dialog box coordinates based on image size
        x = max(0, min(DialogConfig.DIALOG_X, width - DialogConfig.DIALOG_W))
        y = max(0, min(DialogConfig.DIALOG_Y, height - DialogConfig.DIALOG_H))
        w = min(DialogConfig.DIALOG_W, width - x)
        h = min(DialogConfig.DIALOG_H, height - y)
        
        # Check if crop region is valid
        if w <= 0 or h <= 0:
            return {
                'name': '',
                'type': '',
                'rarity': '',
                'stats': {
                    'hp': 0, 'atk': 0, 'def': 0,
                    'crit': 0.0, 'atk_spd': 0.0, 'evasion': 0.0
                },
                'additional_effects': {'effects': []}
            }
        
        # Crop dialog box using adjusted coordinates
        dialog_img = img[y:y+h, x:x+w]
        
        # Save debug images
        base_path = os.path.join("debug_images", os.path.splitext(os.path.basename(image_path))[0])
        
        # Save original image with dialog box outline
        debug_original = img.copy()
        
        # Draw dialog box
        cv2.rectangle(debug_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_original, "Dialog Box", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add grid lines
        grid_color = (128, 128, 128)
        grid_step = 100
        
        # Vertical lines
        for i in range(0, width, grid_step):
            cv2.line(debug_original, (i, 0), (i, height), grid_color, 1)
            cv2.putText(debug_original, str(i), (i, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        # Horizontal lines
        for i in range(0, height, grid_step):
            cv2.line(debug_original, (0, i), (width, i), grid_color, 1)
            cv2.putText(debug_original, str(i), (5, i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        cv2.imwrite(f"{base_path}_original.jpg", debug_original)
        
        # Save cropped dialog box
        dialog_debug_path = f"{base_path}_dialog.jpg"
        cv2.imwrite(dialog_debug_path, dialog_img)
        
        # Create a copy for visualization
        debug_img = dialog_img.copy()
        
        # Draw and save each region
        regions = {
            'name': DialogConfig.NAME_REGION,
            'type': DialogConfig.TYPE_REGION,
            'quality': DialogConfig.QUALITY_REGION,
            'stats': DialogConfig.STATS_REGION,
            'effects': DialogConfig.EFFECTS_REGION
        }
        
        colors = {
            'name': (255, 0, 0),      # Blue
            'type': (0, 255, 0),      # Green
            'quality': (0, 0, 255),   # Red
            'stats': (255, 255, 0),   # Cyan
            'effects': (255, 0, 255)  # Magenta
        }
        
        # Add grid lines to dialog debug image
        dialog_w, dialog_h = dialog_img.shape[1], dialog_img.shape[0]
        grid_step = 50
        
        # Vertical lines
        for i in range(0, dialog_w, grid_step):
            cv2.line(debug_img, (i, 0), (i, dialog_h), grid_color, 1)
            cv2.putText(debug_img, str(i), (i, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        # Horizontal lines
        for i in range(0, dialog_h, grid_step):
            cv2.line(debug_img, (0, i), (dialog_w, i), grid_color, 1)
            cv2.putText(debug_img, str(i), (5, i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        for name, config in regions.items():
            # Adjust region coordinates to fit within dialog box
            rx = max(0, min(config.x, w - config.w))
            ry = max(0, min(config.y, h - config.h))
            rw = min(config.w, w - rx)
            rh = min(config.h, h - ry)
            
            # Skip if region is invalid
            if rw <= 0 or rh <= 0:
                continue
            
            # Draw rectangle on debug image
            cv2.rectangle(
                debug_img,
                (rx, ry),
                (rx + rw, ry + rh),
                colors[name],
                2
            )
            
            # Add region label with coordinates
            label = f"{name} ({rx},{ry})"
            cv2.putText(
                debug_img,
                label,
                (rx, ry - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[name],
                1
            )
            
            # Save individual region
            try:
                region = dialog_img[ry:ry+rh, rx:rx+rw]
                if region.size > 0:
                    region_path = f"{base_path}_{name}.jpg"
                    cv2.imwrite(region_path, region)
            except Exception as e:
                pass
        
        # Save debug image with all regions marked
        debug_path = f"{base_path}_regions.jpg"
        cv2.imwrite(debug_path, debug_img)
        
        # Process each region
        name = OCRService.process_name_region(dialog_img, DialogConfig.NAME_REGION)
        equip_type, quality = OCRService.process_type_and_quality(
            dialog_img, DialogConfig.TYPE_REGION, DialogConfig.QUALITY_REGION
        )
        stats = OCRService.process_stats_region(dialog_img, DialogConfig.STATS_REGION)
        effects = OCRService.process_effects_region(dialog_img, DialogConfig.EFFECTS_REGION)
        return {
            'name': name,
            'type': equip_type,
            'rarity': quality,
            'stats': stats,
            'additional_effects': effects
        }

    @staticmethod
    def find_dialog_corners(image_path: str) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """Find the dialog box using template matching with corner templates."""
        # Read the image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load corner templates
        template_dir = "assets/templates/dialog_corners"
        templates = {
            'top_left': cv2.imread(f"{template_dir}/top_left.png", cv2.IMREAD_GRAYSCALE),
            'top_right': cv2.imread(f"{template_dir}/top_right.png", cv2.IMREAD_GRAYSCALE),
            'bottom_left': cv2.imread(f"{template_dir}/bottom_left.png", cv2.IMREAD_GRAYSCALE),
            'bottom_right': cv2.imread(f"{template_dir}/bottom_right.png", cv2.IMREAD_GRAYSCALE)
        }
        
        # Find corners using template matching
        corners = {}
        for name, template in templates.items():
            # Apply template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Adjust point based on template size
            if name in ['top_right', 'bottom_right']:
                max_loc = (max_loc[0] + template.shape[1], max_loc[1])
            if name in ['bottom_left', 'bottom_right']:
                max_loc = (max_loc[0], max_loc[1] + template.shape[0])
                
            corners[name] = max_loc
        
        # Get bounding rectangle
        x = min(corners['top_left'][0], corners['bottom_left'][0])
        y = min(corners['top_left'][1], corners['top_right'][1])
        w = max(corners['top_right'][0], corners['bottom_right'][0]) - x
        h = max(corners['bottom_left'][1], corners['bottom_right'][1]) - y
        
        # Add padding
        padding = int(min(w, h) * 0.02)  # 2% of min dimension
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        # Draw debug visualization
        # Draw corner points
        cv2.circle(img, corners['top_left'], 5, (255, 0, 0), -1)
        cv2.circle(img, corners['top_right'], 5, (0, 255, 0), -1)
        cv2.circle(img, corners['bottom_left'], 5, (0, 0, 255), -1)
        cv2.circle(img, corners['bottom_right'], 5, (255, 255, 0), -1)
        
        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        return (x, y, w, h), img

    @staticmethod
    def segment_dialog_regions(image: np.ndarray) -> Dict[str, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Segment dialog box into different regions."""
        height, width = image.shape[:2]
        
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define regions based on relative positions
        name_region = (0, int(height * 0.15))  # Top 15%
        info_region = (name_region[1], int(height * 0.3))  # Next 15%
        stats_region = (info_region[1], int(height * 0.7))  # Middle 40%
        effects_region = (stats_region[1], height)  # Bottom 30%
        
        # Extract regions
        regions = {
            'name': (
                image[name_region[0]:name_region[1], :],
                (0, name_region[0], width, name_region[1] - name_region[0])
            ),
            'info': (
                image[info_region[0]:info_region[1], :],
                (0, info_region[0], width, info_region[1] - info_region[0])
            ),
            'stats': (
                image[stats_region[0]:stats_region[1], :],
                (0, stats_region[0], width, stats_region[1] - stats_region[0])
            ),
            'effects': (
                image[effects_region[0]:effects_region[1], :],
                (0, effects_region[0], width, effects_region[1] - effects_region[0])
            )
        }
        
        return regions

    @staticmethod
    def save_debug_image(image_path: str, output_path: str):
        """Save debug image with dialog box highlighted."""
        dialog_rect, img = OCRService.find_dialog_corners(image_path)
        if dialog_rect is not None:
            x, y, w, h = dialog_rect
            # Draw the dialog box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add label
            cv2.putText(img, "Dialog Box", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)

    @staticmethod
    def get_raw_text(image_path: str) -> str:
        """Trả về toàn bộ text OCR thô từ ảnh."""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return ""
        processed = OCRService.preprocess_for_ocr(img)
        text = pytesseract.image_to_string(processed)
        return text 