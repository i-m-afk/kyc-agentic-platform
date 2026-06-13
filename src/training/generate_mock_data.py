import os
import argparse
from PIL import Image, ImageDraw
import random

def get_args():
    parser = argparse.ArgumentParser(description="Generate synthetic/mock face and spoof image dataset for pipeline test runs")
    parser.add_argument("--output_dir", type=str, default="./mock_dataset", help="Directory where dataset splits will be created")
    parser.add_argument("--num_images", type=int, default=15, help="Number of images to generate per subfolder")
    return parser.parse_args()

def generate_dummy_image(is_real: bool, path: str):
    """Creates a synthetic image with distinct patterns for Real vs Spoof."""
    # 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(30, 41, 59))
    draw = ImageDraw.Draw(img)
    
    if is_real:
        # Draw a face-like shape (clean circle in center)
        draw.ellipse([50, 50, 174, 174], fill=(224, 130, 98), outline=(255, 255, 255), width=2)
        # Add eyes
        draw.ellipse([80, 90, 100, 110], fill=(255, 255, 255))
        draw.ellipse([124, 90, 144, 110], fill=(255, 255, 255))
    else:
        # Draw a phone/screen frame shape or scanlines indicating a photo reprint spoof
        draw.rectangle([30, 30, 194, 194], fill=(50, 50, 50), outline=(239, 68, 68), width=3)
        # Add horizontal scanlines
        for y in range(40, 180, 10):
            draw.line([40, y, 184, y], fill=(20, 20, 20), width=1)
            
    img.save(path)

def main():
    args = get_args()
    
    splits = ["train", "val"]
    classes = ["real", "spoof"]
    
    print(f"Generating synthetic liveness dataset at: {args.output_dir}")
    
    for split in splits:
        for cls in classes:
            folder_path = os.path.join(args.output_dir, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            
            is_real = (cls == "real")
            for i in range(args.num_images):
                img_name = f"dummy_{i:03d}.jpg"
                img_path = os.path.join(folder_path, img_name)
                generate_dummy_image(is_real, img_path)
                
    print(f"Generation complete! Synthesized {len(splits) * len(classes) * args.num_images} images.")

if __name__ == "__main__":
    main()
