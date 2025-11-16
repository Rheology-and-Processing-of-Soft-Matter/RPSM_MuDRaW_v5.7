from PIL import Image
import os

def merge_images_x(image_path1, image_path2, output_path="merged_image.png"):
    # Sort paths alphabetically
    paths = sorted([image_path1, image_path2])

    # Open images
    images = [Image.open(p) for p in paths]

    # Determine final canvas height
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images)

    # Create blank canvas (same mode as first image)
    merged = Image.new(images[0].mode, (total_width, max_height))

    # Paste images side by side without resizing
    x_offset = 0
    for img in images:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width

    merged.save(output_path)
    print(f"Merged image saved as: {os.path.abspath(output_path)}")

# Example usage
merge_images_x(
    "/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/Temp/FR/_Old drive/CNC4 PS2/PLI/MVI_3015_st_circular.png",
    "/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/Temp/FR/_Old drive/CNC4 PS2/PLI/MVI_3016_st_circular.png"
)

# Example usage
#merge_images_x("/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/Temp/FR/_Old drive/CNC4 PS2/PLI/MVI_3015_st_circular.png", "/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/Temp/FR/_Old drive/CNC4 PS2/PLI/MVI_3016_st_circular.png")

