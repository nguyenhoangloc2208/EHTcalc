import os
import random
import cv2
import numpy as np
from app.services.ocr import OCRService, DialogConfig
import json

def validate_dialog_box(width, height, x, y, w, h):
    """Validate dialog box dimensions and position."""
    # Minimum size requirements
    MIN_WIDTH = 100
    MIN_HEIGHT = 100
    
    # Aspect ratio limits (width/height)
    MIN_ASPECT_RATIO = 0.4  # Allow slightly wider dialogs
    MAX_ASPECT_RATIO = 2.0  # Allow some flexibility
    
    # Validate coordinates
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    # Check minimum size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return None, f"Dialog box too small: {w}x{h} (minimum: {MIN_WIDTH}x{MIN_HEIGHT})"
    
    # Check aspect ratio
    aspect_ratio = w / h
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None, f"Invalid aspect ratio: {aspect_ratio:.2f} (expected: {MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO})"
    
    return (x, y, w, h), None

def draw_ocr_regions(dialog_img):
    """Draw OCR regions on cropped dialog image."""
    # Define colors for different regions
    colors = {
        'name': (255, 0, 0),     # Blue
        'type': (0, 255, 0),     # Green
        'quality': (0, 0, 255),  # Red
        'stats': (255, 255, 0),  # Cyan
        'effects': (255, 0, 255) # Magenta
    }
    
    # Get dialog config
    regions = {
        'name': DialogConfig.NAME_REGION,
        'type': DialogConfig.TYPE_REGION,
        'quality': DialogConfig.QUALITY_REGION,
        'stats': DialogConfig.STATS_REGION,
        'effects': DialogConfig.EFFECTS_REGION
    }
    
    height, width = dialog_img.shape[:2]
    
    # Draw grid lines
    grid_color = (128, 128, 128)
    grid_step = 100
    
    # Vertical lines
    for i in range(0, width, grid_step):
        cv2.line(dialog_img, (i, 0), (i, height), grid_color, 1)
        cv2.putText(dialog_img, str(i), (i, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
    
    # Horizontal lines
    for i in range(0, height, grid_step):
        cv2.line(dialog_img, (0, i), (width, i), grid_color, 1)
        cv2.putText(dialog_img, str(i), (5, i),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
    
    # Draw each region
    for name, config in regions.items():
        # Use coordinates directly since we're working with cropped image
        x = config.x
        y = config.y
        w = min(config.w, width - x)
        h = min(config.h, height - y)
        
        # Draw rectangle
        cv2.rectangle(dialog_img, (x, y), (x+w, y+h), colors[name], 2)
        
        # Add label with coordinates
        label = f"{name}: ({x},{y})"
        cv2.putText(dialog_img, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)

def process_dialog_with_retry(image_path, max_retries=10):
    """Process dialog with retry mechanism."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image")
                
            height, width = img.shape[:2]
            
            # Try different detection strategies
            strategies = [
                lambda: find_dialog_by_template_matching(img),
                lambda: find_dialog_by_edge_detection(img),
                lambda: find_dialog_by_contours(img)
            ]
            
            strategy_idx = attempt % len(strategies)
            x, y, w, h = strategies[strategy_idx]()
            
            # Validate detected region
            result, error = validate_dialog_box(width, height, x, y, w, h)
            if result is not None:
                x, y, w, h = result
                # Crop dialog
                dialog_img = img[y:y+h, x:x+w].copy()
                
                # Draw OCR regions and grid on cropped image
                draw_ocr_regions(dialog_img)
                
                # Save processed dialog
                debug_dir = "debug_images"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                    
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_path = os.path.join(debug_dir, f"{base_name}_processed.jpg")
                cv2.imwrite(debug_path, dialog_img)
                
                return result
                
            last_error = error
            print(f"Attempt {attempt + 1} failed: {error}")
            
        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempt + 1} failed with error: {e}")
            
    raise ValueError(f"Failed to process dialog after {max_retries} attempts. Last error: {last_error}")

def find_dialog_by_template_matching(img):
    """Find dialog box using template matching."""
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
    
    corners = {}
    for name, template in templates.items():
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        if name in ['top_right', 'bottom_right']:
            max_loc = (max_loc[0] + template.shape[1], max_loc[1])
        if name in ['bottom_left', 'bottom_right']:
            max_loc = (max_loc[0], max_loc[1] + template.shape[0])
            
        corners[name] = max_loc
    
    x = min(corners['top_left'][0], corners['bottom_left'][0])
    y = min(corners['top_left'][1], corners['top_right'][1])
    w = max(corners['top_right'][0], corners['bottom_right'][0]) - x
    h = max(corners['bottom_left'][1], corners['bottom_right'][1]) - y
    
    return x, y, w, h

def find_dialog_by_edge_detection(img):
    """Find dialog box using edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest rectangular contour
    max_area = 0
    best_rect = None
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Use int32 instead of int0
        area = cv2.contourArea(box)
        
        if area > max_area:
            max_area = area
            best_rect = box
    
    if best_rect is None:
        raise ValueError("No dialog box found")
        
    x = int(min(best_rect[:, 0]))
    y = int(min(best_rect[:, 1]))
    w = int(max(best_rect[:, 0]) - x)
    h = int(max(best_rect[:, 1]) - y)
    
    return x, y, w, h

def find_dialog_by_contours(img):
    """Find dialog box using contour detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if not contours:
        raise ValueError("No contours found")
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h

def test_regions():
    # Get all images from the images directory
    image_dir = "images"
    # Only process original images (not debug images)
    image_files = [
        f for f in os.listdir(image_dir) 
        if f.endswith('.jpg') and not any(x in f for x in ['_dialog', '_regions', '_name', '_type', '_quality', '_stats', '_effects'])
    ]
    
    if not image_files:
        print("No images found in the images directory")
        return
    
    # Select random images
    test_images = random.sample(image_files, min(1, len(image_files)))
    
    success_count = 0
    total_count = len(test_images)
    
    for idx, image_file in enumerate(test_images, 1):
        print(f"\n{'='*50}")
        print(f"Testing image {idx}/{total_count}: {image_file}")
        print(f"{'='*50}\n")
        
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image")
                
            height, width = img.shape[:2]
            
            # Try different detection strategies
            strategies = [
                lambda: find_dialog_by_template_matching(img),
                lambda: find_dialog_by_edge_detection(img),
                lambda: find_dialog_by_contours(img)
            ]
            
            last_error = None
            success = False
            
            for attempt in range(10):  # max_retries = 10
                try:
                    strategy_idx = attempt % len(strategies)
                    x, y, w, h = strategies[strategy_idx]()
                    
                    # Validate detected region
                    result, error = validate_dialog_box(width, height, x, y, w, h)
                    if result is not None:
                        x, y, w, h = result
                        # Crop dialog
                        dialog_img = img[y:y+h, x:x+w].copy()
                        
                        # Process OCR for each region
                        print("\nProcessing OCR regions:")
                        print("-" * 30)
                        
                        # Process name
                        name = OCRService.process_name_region(dialog_img, DialogConfig.NAME_REGION)
                        print(f"Name: {name}")
                        
                        # Process type and quality
                        equip_type, quality = OCRService.process_type_and_quality(
                            dialog_img, 
                            DialogConfig.TYPE_REGION,
                            DialogConfig.QUALITY_REGION
                        )
                        print(f"Type: {equip_type}")
                        print(f"Quality: {quality}")
                        
                        # Process stats
                        stats = OCRService.process_stats_region(dialog_img, DialogConfig.STATS_REGION)
                        print("\nStats:")
                        for stat_name, value in stats.items():
                            print(f"  {stat_name.upper()}: {value}")
                        
                        # Process effects
                        effects = OCRService.process_effects_region(dialog_img, DialogConfig.EFFECTS_REGION)
                        print("\nEffects:")
                        print("  Raw effects:")
                        for effect in effects.get('effects', []):
                            print(f"    - {effect}")
                        print("  Parsed effects:")
                        for k, v in effects.get('parsed_effects', {}).items():
                            print(f"    {k}: {v}")
                        print("  Unparsed effects:")
                        for effect in effects.get('unparsed_effects', []):
                            print(f"    - {effect}")
                        print("\nDebug image saved to:")
                        print(f"  debug_images/{os.path.splitext(image_file)[0]}_processed.jpg")
                        print("-" * 30)

                        # Save all data to debug_images/debug_data_1.json
                        debug_data = {
                            'name': name,
                            'type': equip_type,
                            'quality': quality,
                            'stats': stats,
                            'effects': effects
                        }
                        with open('debug_images/debug_data_1.json', 'w', encoding='utf-8') as f:
                            json.dump(debug_data, f, ensure_ascii=False, indent=2)

                        # Draw OCR regions and grid on cropped image
                        draw_ocr_regions(dialog_img)
                        
                        # Save processed dialog
                        debug_dir = "debug_images"
                        if not os.path.exists(debug_dir):
                            os.makedirs(debug_dir)
                            
                        # Use sequential numbering for debug images
                        debug_path = os.path.join(debug_dir, f"debug_image_{idx}.jpg")
                        cv2.imwrite(debug_path, dialog_img)
                        
                        print(f"\nSuccessfully processed {image_file}")
                        print(f"Processed image saved to: {debug_path}")
                        print(f"Dialog box coordinates: x={x}, y={y}, w={w}, h={h}")
                        success = True
                        success_count += 1
                        break
                        
                    last_error = error
                    print(f"Attempt {attempt + 1} failed: {error}")
                    
                except Exception as e:
                    last_error = str(e)
                    print(f"Attempt {attempt + 1} failed with error: {e}")
            
            if not success:
                print(f"Failed to process {image_file} after 10 attempts.")
                print(f"Last error: {last_error}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print(f"\nProcessing complete: {success_count}/{total_count} images successful")

def process_single_image(image_path, max_retries=10):
    """Process a single image and draw dialog box without cropping."""
    print(f"\n{'='*50}")
    print(f"Processing image: {image_path}")
    print(f"{'='*50}\n")
    
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
            
        height, width = img.shape[:2]
        
        # Try different detection strategies
        strategies = [
            lambda: find_dialog_by_template_matching(img),
            lambda: find_dialog_by_edge_detection(img),
            lambda: find_dialog_by_contours(img)
        ]
        
        last_error = None
        success = False
        
        for attempt in range(max_retries):
            try:
                strategy_idx = attempt % len(strategies)
                x, y, w, h = strategies[strategy_idx]()
                
                # Validate detected region
                result, error = validate_dialog_box(width, height, x, y, w, h)
                if result is not None:
                    x, y, w, h = result
                    # Crop dialog
                    dialog_img = img[y:y+h, x:x+w].copy()
                    
                    # Process OCR for each region
                    print("\nProcessing OCR regions:")
                    print("-" * 30)
                    
                    # Process name
                    name = OCRService.process_name_region(dialog_img, DialogConfig.NAME_REGION)
                    print(f"Name: {name}")
                    
                    # Process type and quality
                    equip_type, quality = OCRService.process_type_and_quality(
                        dialog_img, 
                        DialogConfig.TYPE_REGION,
                        DialogConfig.QUALITY_REGION
                    )
                    print(f"Type: {equip_type}")
                    print(f"Quality: {quality}")
                    
                    # Process stats
                    stats = OCRService.process_stats_region(dialog_img, DialogConfig.STATS_REGION)
                    print("\nStats:")
                    for stat_name, value in stats.items():
                        print(f"  {stat_name.upper()}: {value}")
                    
                    # Process effects
                    effects = OCRService.process_effects_region(dialog_img, DialogConfig.EFFECTS_REGION)
                    print("\nEffects:")
                    print("  Raw effects:")
                    for effect in effects.get('effects', []):
                        print(f"    - {effect}")
                    print("  Parsed effects:")
                    for k, v in effects.get('parsed_effects', {}).items():
                        print(f"    {k}: {v}")
                    print("  Unparsed effects:")
                    for effect in effects.get('unparsed_effects', []):
                        print(f"    - {effect}")
                    print("\nDebug image saved to:")
                    print(f"  debug_images/{os.path.splitext(image_file)[0]}_processed.jpg")
                    print("-" * 30)

                    # Save all data to debug_images/debug_data_1.json
                    debug_data = {
                        'name': name,
                        'type': equip_type,
                        'quality': quality,
                        'stats': stats,
                        'effects': effects
                    }
                    with open('debug_images/debug_data_1.json', 'w', encoding='utf-8') as f:
                        json.dump(debug_data, f, ensure_ascii=False, indent=2)

                    # Draw OCR regions and grid on cropped image
                    draw_ocr_regions(dialog_img)
                    
                    # Save processed dialog
                    debug_dir = "debug_images"
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                        
                    # Use sequential numbering for debug images
                    debug_path = os.path.join(debug_dir, f"debug_image_1.jpg")  # Single image always gets number 1
                    cv2.imwrite(debug_path, dialog_img)
                    
                    print(f"\nSuccessfully processed image")
                    print(f"Processed image saved to: {debug_path}")
                    print(f"Dialog box coordinates: x={x}, y={y}, w={w}, h={h}")
                    success = True
                    break
                    
                last_error = error
                print(f"Attempt {attempt + 1} failed: {error}")
                
            except Exception as e:
                last_error = str(e)
                print(f"Attempt {attempt + 1} failed with error: {e}")
        
        if not success:
            print(f"Failed to process image after {max_retries} attempts.")
            print(f"Last error: {last_error}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process dialog boxes in images')
    parser.add_argument('--image', type=str, help='Path to single image to process')
    args = parser.parse_args()
    
    if args.image:
        process_single_image(args.image)
    else:
        test_regions() 