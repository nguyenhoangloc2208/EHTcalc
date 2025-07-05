import os
import random
import shutil
from app.services.ocr import OCRService

# Create test directory if it doesn't exist
test_dir = 'test_images'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get list of images
image_dir = 'images'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Select 5 random images
selected_images = random.sample(image_files, 5)

print("Selected images for testing:")
for i, image in enumerate(selected_images, 1):
    print(f"{i}. {image}")
    
    # Copy image to test directory
    src_path = os.path.join(image_dir, image)
    test_path = os.path.join(test_dir, image)
    shutil.copy2(src_path, test_path)
    
    print("\nProcessing image...")
    try:
        # Extract stats using OCR
        stats = OCRService.extract_stats(test_path)
        
        # Print results
        print("\nExtracted stats:")
        print(f"Name: {stats['name']}")
        print(f"Type: {stats['type']}")
        print(f"Rarity: {stats['rarity']}")
        print("\nBase stats:")
        for stat, value in stats['stats'].items():
            print(f"{stat.upper()}: {value}")
        
        if stats['additional_effects']:
            print("\nAdditional effects:")
            for effect in stats['additional_effects'].get('effects', []):
                print(f"- {effect}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    
    print("\n" + "="*50 + "\n") 