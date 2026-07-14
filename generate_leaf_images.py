"""
Procedurally generate a synthetic leaf-image dataset for 3 classes:
Healthy, Leaf Blight, Leaf Rust.
Images encode class-characteristic color/texture signatures so a
downstream classifier can learn a genuine (if simplified) decision boundary.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

np.random.seed(7)
OUT = "/home/claude/crop_project/data/leaf_images"
os.makedirs(OUT, exist_ok=True)

SIZE = 96
N_PER_CLASS = 120

def base_leaf(hue_shift=0):
    img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    green = (34 + hue_shift, 139, 34)
    draw.ellipse([10, 6, 86, 90], fill=green)
    draw.line([48, 10, 48, 86], fill=(20, 90, 20), width=2)
    for y in range(20, 80, 10):
        draw.line([48, y, 20 + (y % 20), y + 8], fill=(20, 90, 20), width=1)
        draw.line([48, y, 76 - (y % 20), y + 8], fill=(20, 90, 20), width=1)
    return img

def add_noise(img, sigma=6):
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def make_healthy():
    img = base_leaf(hue_shift=np.random.randint(-8, 8))
    # occasionally add a couple of faint natural blemishes so classes aren't perfectly separable
    if np.random.rand() < 0.25:
        draw = ImageDraw.Draw(img)
        for _ in range(np.random.randint(1, 3)):
            cx, cy = np.random.randint(20, 76, size=2)
            r = np.random.randint(1, 3)
            faint = (120, 90, 40)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=faint)
    return add_noise(img, sigma=8)

def make_blight():
    # brown irregular necrotic patches (some images have few/faint spots to overlap with healthy/rust)
    img = base_leaf(hue_shift=np.random.randint(-10, 0))
    draw = ImageDraw.Draw(img)
    n_spots = np.random.randint(2, 8)
    for _ in range(n_spots):
        cx, cy = np.random.randint(20, 76, size=2)
        r = np.random.randint(2, 9)
        brown = (101 + np.random.randint(-20,20), 67 + np.random.randint(-20,20), 33 + np.random.randint(-10,20))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=brown)
    img = img.filter(ImageFilter.GaussianBlur(np.random.uniform(0.4, 1.2)))
    return add_noise(img, sigma=9)

def make_rust():
    # small orange-red pustules scattered, variable density (some sparse, overlapping blight range)
    img = base_leaf(hue_shift=np.random.randint(0, 10))
    draw = ImageDraw.Draw(img)
    n_spots = np.random.randint(10, 45)
    for _ in range(n_spots):
        cx, cy = np.random.randint(15, 82, size=2)
        r = np.random.randint(1, 4)
        orange = (204 + np.random.randint(-25,25), 102 + np.random.randint(-25,25), 0 + np.random.randint(0,20))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=orange)
    return add_noise(img, sigma=8)

generators = {"healthy": make_healthy, "blight": make_blight, "rust": make_rust}

manifest = []
for cls, gen in generators.items():
    cdir = os.path.join(OUT, cls)
    os.makedirs(cdir, exist_ok=True)
    for i in range(N_PER_CLASS):
        img = gen()
        path = os.path.join(cdir, f"{cls}_{i:03d}.png")
        img.save(path)
        manifest.append((path, cls))

print(f"Generated {len(manifest)} images across {len(generators)} classes in {OUT}")
