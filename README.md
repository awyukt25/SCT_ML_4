# Hand Gesture Recognition using a Residual CNN

A convolutional neural network that classifies 10 different hand gestures from images, built on the Kaggle **LeapGestRecog** dataset. The goal is a model that can power intuitive, gesture-based control systems and that is evaluated honestly, on people it has never seen.

## Overview

| | |
|---|---|
| **Task** | Multi-class image classification (10 hand gestures) |
| **Approach** | Deep learning: small residual CNN trained from scratch |
| **Framework** | TensorFlow / Keras (`tf.data` pipeline, Functional API) |
| **Dataset** | [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog): 20,000 infrared images, 10 subjects, 10 gestures |
| **Input** | 64×64 grayscale images |

### Gesture classes

`01_palm`, `02_l`, `03_fist`, `04_fist_moved`, `05_thumb`, `06_index`, `07_ok`, `08_palm_moved`, `09_c`, `10_down`

Class names are read from the dataset's folder structure rather than hardcoded.

## Why a Subject-Based Split

The dataset consists of video frames, so neighbouring frames of the same person are nearly identical. A normal random train/test split puts near-duplicates of the same frame on both sides, which inflates accuracy to almost 100% without proving the model works on a new user.

This project therefore splits by **person**:

| Set | Subjects |
|---|---|
| Train | 6 |
| Validation | 2 |
| Test | 2 |

No subject appears in more than one set, including validation, so the training curves and the final test score both reflect performance on unseen people. An optional experiment in the notebook trains a second model on a random split so you can measure the inflation directly.


The dataset is downloaded automatically with `kagglehub`, so no manual upload is needed.

## Workflow

1. **Setup:** imports, random seeds, and config switches.
2. **Download:** fetch LeapGestRecog through `kagglehub`.
3. **Index files:** build a table of *(path, label, subject)* and check the frame counts per subject and gesture. Preview one example of every gesture.
4. **Load images:** read as grayscale, resize to 64×64, and store as `uint8` to save memory. Unreadable files are dropped.
5. **Subject-based split:** train / validation / test sets with no shared people.
6. **`tf.data` pipelines:** batching, shuffling, and prefetching.
7. **Model:** a residual CNN with three stages (32 → 64 → 128 filters), batch normalization, spatial dropout, global average pooling, and a dense head. Rescaling and augmentation (small rotation, shift, and zoom) are built into the model and only active during training.
8. **Training:** Adam optimizer, label smoothing, `EarlyStopping`, and `ReduceLROnPlateau`.
9. **Evaluation on unseen people:** accuracy, per-class classification report, normalized confusion matrix, a random sample of test predictions (correct in green, wrong in red), and a gallery of the model's most confident mistakes.
10. **Optional: leakage demo** (`RUN_LEAKAGE_DEMO`): compares the subject-based score to a random-split score.
11. **Optional: subject-wise cross-validation** (`RUN_GROUP_CV`): `GroupKFold` where every fold holds out entire people.
12. **Save and predict:** the model is saved as `gesture_model.keras`, and `predict_gesture(path)` classifies a new image with its top-3 probabilities.

## Results

Fill in with your actual run output from Colab:

| Metric | Score |
|---|---|
| Test accuracy (unseen subjects) | 0.9550 |


Report the **unseen-subject** number as the real performance. The random-split figure is only there to show how leakage inflates results.

## Limitations

- The dataset has only 10 people in a controlled setting, so results may not transfer to other cameras, lighting, backgrounds, or hand shapes.
- Images are infrared Leap Motion captures. A model trained on them will not work directly on ordinary RGB webcam footage without retraining or fine-tuning.
- Classification is per single frame. Recognizing dynamic gestures (motion over time) would need a sequence model or temporal smoothing.
- Accuracy on unseen people varies with which subjects fall in the test set, which is why the optional cross-validation is worth running.

## How to Run

1. Open `hand_gesture_recognition_cnn.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells top to bottom. The dataset is about 2 GB, so the first download takes a couple of minutes.
4. To run the optional experiments, set `RUN_LEAKAGE_DEMO` and/or `RUN_GROUP_CV` to `True` in the setup cell. They train extra models and add meaningful runtime.
5. Download `gesture_model.keras` from the Colab file browser if you want to reuse the trained model.

## Requirements

```
tensorflow
kagglehub
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install tensorflow kagglehub opencv-python numpy pandas matplotlib seaborn scikit-learn
```

(Google Colab already includes most of these; `kagglehub` may need installing.)
