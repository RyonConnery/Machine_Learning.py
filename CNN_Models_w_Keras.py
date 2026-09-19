# ============================================================
# CNN Models Using Keras
# ============================================================

# Import the libraries required to load Fashion MNIST,
# create representative stratified subsets, construct the
# three required CNN architectures, train them, evaluate
# test performance, and compare the results.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import fashion_mnist


# ============================================================
# 1. MAKE THE EXPERIMENT REPRODUCIBLE
# ============================================================

# Set random seeds so the sampling and model initialization are
# more consistent if the notebook is run again.
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD THE FASHION MNIST DATASET
# ============================================================

# Fashion MNIST contains:
# 60,000 training images
# 10,000 test images
# 10 clothing categories
# Each grayscale image is 28 x 28 pixels.
(train_images_full, train_labels_full), (
    test_images_full,
    test_labels_full
) = fashion_mnist.load_data()

print("FASHION MNIST DATASET LOADED SUCCESSFULLY")
print("-" * 70)

print(
    f"Original training images: "
    f"{train_images_full.shape[0]:,}"
)

print(
    f"Original test images: "
    f"{test_images_full.shape[0]:,}"
)

print(
    f"Original image dimensions: "
    f"{train_images_full.shape[1:]}"
)

print(
    f"Number of classes: "
    f"{len(np.unique(train_labels_full))}"
)


# ============================================================
# 3. CREATE REPRESENTATIVE STRATIFIED SUBSETS
# ============================================================

# Training all three CNNs for 10 epochs on the entire 60,000-image
# dataset is unnecessarily expensive for this architecture-comparison
# exercise.
#
# A stratified subset is used so every Fashion MNIST class remains
# proportionally represented.
#
# 6,000 training images = approximately 600 examples per class.
# 1,000 test images = approximately 100 examples per class.

train_images, _, train_labels, _ = train_test_split(
    train_images_full,
    train_labels_full,
    train_size=6000,
    random_state=42,
    stratify=train_labels_full
)

test_images, _, test_labels, _ = train_test_split(
    test_images_full,
    test_labels_full,
    train_size=1000,
    random_state=42,
    stratify=test_labels_full
)

print("\nREPRESENTATIVE DATA SUBSETS")
print("-" * 70)

print(
    f"CNN training images: "
    f"{train_images.shape[0]:,}"
)

print(
    f"CNN testing images: "
    f"{test_images.shape[0]:,}"
)

print("\nTraining examples per class:")

unique_classes, class_counts = np.unique(
    train_labels,
    return_counts=True
)

for class_value, class_count in zip(
    unique_classes,
    class_counts
):
    print(
        f"Class {class_value}: "
        f"{class_count} images"
    )


# ============================================================
# 4. PREPROCESS THE IMAGES
# ============================================================

# CNN layers require an explicit image-channel dimension.
# Because Fashion MNIST images are grayscale, each image has
# one channel.
#
# Pixel values are also normalized from the original 0-255
# range to 0-1 for stable neural-network training.
train_images = (
    train_images
    .reshape((-1, 28, 28, 1))
    .astype("float32")
    / 255.0
)

test_images = (
    test_images
    .reshape((-1, 28, 28, 1))
    .astype("float32")
    / 255.0
)

print("\nIMAGE PREPROCESSING")
print("-" * 70)

print(
    f"Training image shape: "
    f"{train_images.shape}"
)

print(
    f"Test image shape: "
    f"{test_images.shape}"
)

print(
    f"Normalized pixel range: "
    f"{train_images.min():.1f} to "
    f"{train_images.max():.1f}"
)


# ============================================================
# 5. BUILD MODEL 1 — BASIC CNN
# ============================================================

# Model 1 follows Capella's required basic architecture:
# 32 convolutional filters
# max pooling
# 64 convolutional filters
# max pooling
# 128-neuron dense layer
# 10-class softmax output
#
# ReLU is used in the convolutional and hidden dense layers.
model_1 = models.Sequential([
    layers.Input(
        shape=(28, 28, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

print("\nMODEL 1 — BASIC CNN")
print("-" * 70)
model_1.summary()


# ============================================================
# 6. BUILD MODEL 2 — INCREASED FILTERS
# ============================================================

# Model 2 increases convolutional capacity:
# first convolution = 64 filters
# second convolution = 128 filters
#
# All other major settings remain comparable to Model 1.
model_2 = models.Sequential([
    layers.Input(
        shape=(28, 28, 1)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

print("\nMODEL 2 — INCREASED FILTERS")
print("-" * 70)
model_2.summary()


# ============================================================
# 7. BUILD MODEL 3 — SIGMOID ACTIVATION
# ============================================================

# Model 3 returns to the 32/64-filter structure of Model 1,
# but replaces ReLU with sigmoid activation.
#
# This allows activation-function choice to be compared while
# keeping the primary convolutional filter structure the same.
model_3 = models.Sequential([
    layers.Input(
        shape=(28, 28, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="sigmoid"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="sigmoid"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="sigmoid"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

print("\nMODEL 3 — SIGMOID ACTIVATION")
print("-" * 70)
model_3.summary()


# ============================================================
# 8. COMPILE ALL THREE MODELS
# ============================================================

# All three models use the same optimizer, loss function,
# and accuracy metric so the architecture comparison remains fair.
for model in [
    model_1,
    model_2,
    model_3
]:

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

print(
    "\nALL THREE CNN MODELS "
    "COMPILED SUCCESSFULLY"
)


# ============================================================
# 9. TRAIN MODEL 1 — BASIC CNN
# ============================================================

# Capella requires 10 epochs and a 20% validation split.
# batch_size=128 improves execution efficiency without changing
# the required number of epochs or validation proportion.
print("\nTRAINING MODEL 1 — BASIC CNN")
print("-" * 70)

history_1 = model_1.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.2,
    batch_size=128,
    verbose=2
)


# ============================================================
# 10. TRAIN MODEL 2 — INCREASED FILTERS
# ============================================================

print("\nTRAINING MODEL 2 — INCREASED FILTERS")
print("-" * 70)

history_2 = model_2.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.2,
    batch_size=128,
    verbose=2
)


# ============================================================
# 11. TRAIN MODEL 3 — SIGMOID ACTIVATION
# ============================================================

print("\nTRAINING MODEL 3 — SIGMOID ACTIVATION")
print("-" * 70)

history_3 = model_3.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.2,
    batch_size=128,
    verbose=2
)


# ============================================================
# 12. EVALUATE ALL THREE MODELS
# ============================================================

# Test all three CNNs on the exact same 1,000-image test subset.
# This creates a fair architecture comparison.
test_loss_1, test_acc_1 = model_1.evaluate(
    test_images,
    test_labels,
    verbose=0
)

test_loss_2, test_acc_2 = model_2.evaluate(
    test_images,
    test_labels,
    verbose=0
)

test_loss_3, test_acc_3 = model_3.evaluate(
    test_images,
    test_labels,
    verbose=0
)

print("\nTEST SET RESULTS")
print("-" * 70)

print(
    f"Model 1 — Basic CNN"
    f"\nTest Loss: {test_loss_1:.4f}"
    f"\nTest Accuracy: {test_acc_1:.4f}"
)

print()

print(
    f"Model 2 — Increased Filters"
    f"\nTest Loss: {test_loss_2:.4f}"
    f"\nTest Accuracy: {test_acc_2:.4f}"
)

print()

print(
    f"Model 3 — Sigmoid Activation"
    f"\nTest Loss: {test_loss_3:.4f}"
    f"\nTest Accuracy: {test_acc_3:.4f}"
)


# ============================================================
# 13. CREATE PERFORMANCE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Model 1 — Basic CNN",
        "Model 2 — Increased Filters",
        "Model 3 — Sigmoid Activation"
    ],

    "Architecture": [
        "32/64 filters, ReLU",
        "64/128 filters, ReLU",
        "32/64 filters, Sigmoid"
    ],

    "Test Loss": [
        test_loss_1,
        test_loss_2,
        test_loss_3
    ],

    "Test Accuracy": [
        test_acc_1,
        test_acc_2,
        test_acc_3
    ]
})

comparison_df = comparison_df.sort_values(
    by="Test Accuracy",
    ascending=False
).reset_index(drop=True)

print("\nCNN PERFORMANCE COMPARISON")
print("-" * 70)

display(
    comparison_df.style.format({
        "Test Loss": "{:.4f}",
        "Test Accuracy": "{:.4f}"
    })
)


# ============================================================
# 14. VISUALIZE TEST ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    [
        "Model 1\nBasic CNN",
        "Model 2\nIncreased Filters",
        "Model 3\nSigmoid"
    ],
    [
        test_acc_1,
        test_acc_2,
        test_acc_3
    ]
)

plt.title(
    "Fashion MNIST CNN Test Accuracy"
)

plt.ylabel(
    "Test Accuracy"
)

plt.ylim(
    0,
    1.0
)

plt.tight_layout()
plt.show()


# ============================================================
# 15. COMPARE VALIDATION ACCURACY ACROSS 10 EPOCHS
# ============================================================

# Compare validation performance during training to observe
# how quickly and how consistently each architecture learned.
plt.figure(
    figsize=(9, 6)
)

epochs = range(
    1,
    11
)

plt.plot(
    epochs,
    history_1.history["val_accuracy"],
    marker="o",
    label="Model 1 — Basic CNN"
)

plt.plot(
    epochs,
    history_2.history["val_accuracy"],
    marker="o",
    label="Model 2 — Increased Filters"
)

plt.plot(
    epochs,
    history_3.history["val_accuracy"],
    marker="o",
    label="Model 3 — Sigmoid"
)

plt.title(
    "Validation Accuracy Across 10 Epochs"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Accuracy"
)

plt.xticks(
    epochs
)

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 16. IDENTIFY THE BEST MODEL
# ============================================================

model_names = [
    "Model 1 — Basic CNN",
    "Model 2 — Increased Filters",
    "Model 3 — Sigmoid Activation"
]

test_accuracies = [
    test_acc_1,
    test_acc_2,
    test_acc_3
]

best_index = int(
    np.argmax(test_accuracies)
)

best_model_name = model_names[
    best_index
]

best_accuracy = test_accuracies[
    best_index
]


# ============================================================
# 17. FINAL KEY FINDINGS
# ============================================================

print("\nKEY FINDINGS")
print("-" * 70)

print(
    f"1. Model 1 — Basic CNN test accuracy: "
    f"{test_acc_1:.4f}."
)

print(
    f"2. Model 2 — Increased Filters test accuracy: "
    f"{test_acc_2:.4f}."
)

print(
    f"3. Model 3 — Sigmoid Activation test accuracy: "
    f"{test_acc_3:.4f}."
)

print(
    f"4. The highest-performing architecture was "
    f"{best_model_name}, with test accuracy "
    f"{best_accuracy:.4f}."
)

print(
    "5. Model 2 increased convolutional capacity from "
    "32/64 filters to 64/128 filters."
)

print(
    "6. Model 3 used the same basic 32/64 filter structure "
    "as Model 1 but replaced ReLU with sigmoid activation."
)

print(
    "7. All three models used the same Fashion MNIST subset, "
    "10 training epochs, 20% validation split, Adam optimizer, "
    "sparse categorical crossentropy loss, and test set."
)

print(
    "8. The representative dataset preserved all 10 Fashion "
    "MNIST classes while making the required three-model CNN "
    "comparison practical in the notebook environment."
)

print(
    "\nAssessment 10 CNN comparison completed successfully."
)
