"""Load the trained model and predict a digit from an image file.

Usage: python predict.py path/to/digit.png
"""

import sys

import numpy as np
from PIL import Image, ImageFilter
from tensorflow import keras
import matplotlib.pyplot as plt


def connected_components(mask):
    """Split a boolean mask into a list of 4-connected pixel-coordinate blobs."""
    visited = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    components = []
    ys, xs = np.where(mask)
    for start in zip(ys.tolist(), xs.tolist()):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components


def select_digit_component(mask):
    """Pick the blob that looks like a handwritten stroke, not a page edge/shadow line.

    A photo's shadow edges or page creases can trigger the same local-contrast
    test as ink and end up as separate blobs. Those tend to be long, flat lines,
    while a digit stroke is roughly as tall as it is wide (or taller) -- so
    prefer blobs with plausible proportions before falling back to sheer size.
    """
    components = connected_components(mask)

    def bbox(component):
        ys = [y for y, _ in component]
        xs = [x for _, x in component]
        return min(ys), max(ys), min(xs), max(xs)

    def is_digit_shaped(component):
        top, bottom, left, right = bbox(component)
        height = bottom - top + 1
        width = right - left + 1
        return width <= height * 2.5  # reject wide, flat line-like artifacts

    plausible = [c for c in components if is_digit_shaped(c)]
    pool = plausible if plausible else components
    best = max(pool, key=len)

    result = np.zeros_like(mask)
    ys, xs = zip(*best)
    result[list(ys), list(xs)] = True
    return result


def preprocess(image_path):
    """Turn a photo/scan of a handwritten digit into MNIST-format input."""
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)

    # Estimate the local background brightness by blurring away the thin ink
    # strokes. Comparing each pixel to its own blurred neighborhood -- rather
    # than one global threshold -- keeps this robust to shadows and uneven
    # lighting across a real photo of paper.
    blur_radius = max(1, min(arr.shape) // 20)
    background = np.array(img.filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32)
    diff = background - arr  # positive where a pixel is darker than its surroundings

    # Ink can be dark-on-light or light-on-dark; take whichever direction
    # actually stands out from the background.
    dark_ink = diff > 25
    light_ink = diff < -25
    ink_mask = dark_ink if dark_ink.sum() >= light_ink.sum() else light_ink

    if not ink_mask.any():
        raise ValueError(f"No digit strokes detected in {image_path}")
    ink_mask = select_digit_component(ink_mask)

    # MNIST convention: bright digit on black background. A clean binary mask,
    # resized down with anti-aliasing below, gives soft MNIST-like edges.
    display = np.where(ink_mask, 255, 0).astype(np.uint8)

    # Crop to the digit's bounding box so it isn't tiny/off-center in the frame,
    # matching how MNIST digits are framed.
    rows = ink_mask.any(axis=1)
    cols = ink_mask.any(axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    display = display[top:bottom + 1, left:right + 1]

    # Resize the cropped digit to fit in a 20x20 box (preserving aspect ratio),
    # then paste it centered on a 28x28 canvas -- the same convention MNIST uses.
    digit = Image.fromarray(display)
    digit.thumbnail((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - digit.width) // 2
    paste_y = (28 - digit.height) // 2
    canvas.paste(digit, (paste_x, paste_y))

    normalized = np.array(canvas).astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1), canvas


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/digit.png")
        sys.exit(1)

    image_path = sys.argv[1]
    model = keras.models.load_model("model.keras")

    x, display_img = preprocess(image_path)
    prediction = model.predict(x)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    print(f"Predicted digit: {predicted_digit} (confidence: {confidence:.2%})")

    plt.imshow(display_img, cmap="gray")
    plt.title(f"Predicted: {predicted_digit} ({confidence:.1%})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
