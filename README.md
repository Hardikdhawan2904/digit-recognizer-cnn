# Digit Recognizer CNN

A simple handwritten digit recognizer trained on MNIST, with a script to classify your own handwritten digit photos and a small web UI to try it in the browser.

## Project structure

```
train.py           # trains the CNN on MNIST and saves model.keras
predict.py         # classifies a single image from the command line
app.py             # Flask web UI: upload a photo, get a prediction
requirements.txt   # dependencies
my_digits/          # example handwritten digit photos
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

Loads MNIST, trains a small CNN for 10 epochs, evaluates on the test set, and saves:
- `model.keras` — the trained model
- `training_curves.png` — training/validation accuracy and loss curves

### Model architecture

A minimal CNN, no dropout/batchnorm/regularization:

```
Conv2D(32, 3x3, relu) -> MaxPooling2D
Conv2D(64, 3x3, relu) -> MaxPooling2D
Flatten
Dense(64, relu)
Dense(10, softmax)
```

Typical results: ~99% train accuracy, ~99% validation accuracy, ~98.8% test accuracy.

## Predict from the command line

```bash
python predict.py path/to/digit.jpg
```

Prints the predicted digit and confidence, and shows the image alongside the prediction.

### Preprocessing (why it works on real photos, not just clean scans)

Real photos have uneven lighting, shadows, and gray paper backgrounds — very different from MNIST's clean black-and-white images. `predict.py` handles this by:

1. Converting to grayscale.
2. Comparing each pixel to its *local* blurred neighborhood (not one global brightness cutoff) to find ink strokes, so shadows and lighting gradients don't confuse it.
3. Picking the largest connected blob shaped like a digit stroke, to ignore page edges/creases that can also show up as high-contrast regions.
4. Cropping to the digit's bounding box and centering it in a 28x28 frame, matching MNIST's own convention.

## Web UI

```bash
python app.py
```

Open `http://127.0.0.1:5000`, upload a photo, and click Predict. Shows the predicted digit, confidence, and the actual 28x28 image the model saw.

## Notes

- This is intentionally a simple baseline (no hyperparameter tuning, no advanced regularization) — accuracy on your own handwriting will vary with pen thickness, writing style, and image quality.
