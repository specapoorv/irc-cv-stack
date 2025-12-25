import cv2
import os

# Input and output directories
input_dir = "/home/specapoorv/irc-cv-stack/data/no_augmentation_simclr/orange"
output_dir = "/home/specapoorv/irc-cv-stack/data/no_augmentation_simclr/orange_cropped"
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Loop through all image files
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith(valid_exts):
        continue  # skip non-image files

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping {filename} — could not read.")
        continue

    h, w, _ = img.shape

    if w != 960:
        print(f"⚠️ Skipping {filename} — expected width 4416, got {w}")
        continue

    mid = w // 2
    left_img = img[:, :mid]
    right_img = img[:, mid:]

    # Create new filenames
    name, ext = os.path.splitext(filename)
    left_path = os.path.join(output_dir, f"{name}_left{ext}")
    right_path = os.path.join(output_dir, f"{name}_right{ext}")

    # Save results
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"✅ Split {filename} → {name}_left{ext}, {name}_right{ext}")

print("🎉 Done! All images processed and saved in:", output_dir)
