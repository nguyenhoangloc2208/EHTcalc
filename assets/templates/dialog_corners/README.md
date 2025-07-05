# Dialog Corner Templates

This directory contains template images for detecting dialog box corners in equipment screenshots.

## Files

- `top_left.png`: Template for the top-left corner of the dialog box
- `top_right.png`: Template for the top-right corner of the dialog box
- `bottom_left.png`: Template for the bottom-left corner of the dialog box
- `bottom_right.png`: Template for the bottom-right corner of the dialog box

## Usage

These templates are used by the OCRService to detect the exact position of the dialog box in equipment screenshots using template matching. The templates should be:

1. Clear and sharp images of each corner
2. Cropped to include only the corner pattern
3. Consistent in size (recommended 20x20 pixels)
4. Free from background noise

## Updating Templates

If you need to update these templates:

1. Take a clear screenshot of an equipment dialog box
2. Crop out each corner carefully
3. Save with the same names to maintain compatibility
4. Test thoroughly with various screenshots to ensure accuracy 