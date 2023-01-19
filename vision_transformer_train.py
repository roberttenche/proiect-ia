# tutorial https://keras.io/examples/vision/object_detection_using_vision_transformer/
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.patches as mat_patches
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io
import shutil
import xml.etree.ElementTree as ET
import random

classes = {
    'cat':1,
    'dog':2
}

def parse_data():
    cat_ann_folder = 'data/train/cat_label'
    dog_ann_folder = 'data/train/dog_label'

    test_cat_folder = 'data/test/cat'
    test_dog_folder = 'data/test/dog'

    cat_ann_paths = [
        os.path.join(cat_ann_folder, f) for f in os.listdir(cat_ann_folder) if os.path.isfile(os.path.join(cat_ann_folder, f))
    ]
    dog_ann_paths = [
        os.path.join(dog_ann_folder, f) for f in os.listdir(dog_ann_folder) if os.path.isfile(os.path.join(dog_ann_folder, f))
    ]

    test_cat_paths = [
        os.path.join(test_cat_folder, f) for f in os.listdir(test_cat_folder) if os.path.isfile(os.path.join(test_cat_folder, f))
    ]
    test_dog_paths = [
        os.path.join(test_dog_folder, f) for f in os.listdir(test_dog_folder) if os.path.isfile(os.path.join(test_dog_folder, f))
    ]

    # mix 'em up
    ann_paths = cat_ann_paths + dog_ann_paths
    test_paths = test_cat_paths + test_dog_paths
    random.shuffle(ann_paths)

    image_resize = 224

    images, targets = [], []

    test_images, test_targets = [], []

    for annot in ann_paths:
        xml = ET.parse(annot) 
        root = xml.getroot() # get root object

        img_path = root.find("path").text

        # parse xml annotations
        bbox_coordinates = []
        for member in root.findall('object'):
            class_name = member[0].text # class name
            # bbox coordinates
            top_left_x = int(member[4][0].text)
            top_left_y = int(member[4][1].text)
            bottom_right_x = int(member[4][2].text)
            bottom_right_y = int(member[4][3].text)
            # store data in list
            bbox_coordinates.append([class_name, top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        image = keras.utils.load_img(img_path)
        (w, h) = image.size[:2]

        image = image.resize((image_resize, image_resize))
        images.append(keras.utils.img_to_array(image))

        # apply relative scaling to bounding boxes as per given image and append to list
        targets.append(
            (
                classes[class_name],
                float(top_left_x) / w,
                float(top_left_y) / h,
                float(bottom_right_x) / w,
                float(bottom_right_y) / h,
            )
        )

    for annot in test_paths:
        xml = ET.parse(annot) 
        root = xml.getroot() # get root object

        img_path = root.find("path").text

        # parse xml annotations
        bbox_coordinates = []
        for member in root.findall('object'):
            class_name = member[0].text # class name
            # bbox coordinates
            top_left_x = int(member[4][0].text)
            top_left_y = int(member[4][1].text)
            bottom_right_x = int(member[4][2].text)
            bottom_right_y = int(member[4][3].text)
            # store data in list
            bbox_coordinates.append([class_name, top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        image = keras.utils.load_img(img_path)
        (w, h) = image.size[:2]

        image = image.resize((image_resize, image_resize))
        test_images.append(keras.utils.img_to_array(image))

        # apply relative scaling to bounding boxes as per given image and append to list
        test_targets.append(
            (
                classes[class_name],
                float(top_left_x) / w,
                float(top_left_y) / h,
                float(bottom_right_x) / w,
                float(bottom_right_y) / h,
            )
        )

    return np.asarray(images), np.asarray(targets), np.asarray(test_images), np.asarray(test_targets)

def bounding_box_intersection_over_union(box_predicted, box_truth):
    # get (x, y) coordinates of intersection of bounding boxes
    top_x_intersect = max(box_predicted[0], box_truth[0])
    top_y_intersect = max(box_predicted[1], box_truth[1])
    bottom_x_intersect = min(box_predicted[2], box_truth[2])
    bottom_y_intersect = min(box_predicted[3], box_truth[3])

    # calculate area of the intersection bb (bounding box)
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    )

    # calculate area of the prediction bb and ground-truth bb
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (
        box_predicted[3] - box_predicted[1] + 1
    )
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (
        box_truth[3] - box_truth[1] + 1
    )

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb and ground truth
    # bb areas subtracted by  the interesection area

    # return ioU
    return intersection_area / float(
        box_predicted_area + box_truth_area - intersection_area
    )

# multilayer-perceptron example used https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = layers.Dense(5)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)

def train_model(model, learning_rate, weight_decay, batch_size, num_epochs):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "logs/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            keras.callbacks.EarlyStopping(monitor="accuracy", patience=10),
        ],
    )

    return history

### ### ### ### ### ###
### APP STARTS HERE ###
### ### ### ### ### ###

x_train = []
y_train = []

image_size = 224
patch_size = 32
input_shape = (image_size, image_size, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 25
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers

x_train, y_train, x_test, y_test = parse_data()

plt.figure(figsize=(4, 4))
plt.imshow(x_train[0].astype("uint8"))
plt.axis("off")

patches = Patches(patch_size)(tf.convert_to_tensor([x_train[0]]))

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

history = train_model(vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs)

vit_object_detector.save("vit_object_detector.h5", save_format="h5")

i, mean_iou = 0, 0

# Compare results for 10 images in the test set
for input_image in x_test[:10]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = input_image

    # Display the image
    ax1.imshow(im.astype("uint8"))
    ax2.imshow(im.astype("uint8"))

    input_image = cv2.resize(
        input_image, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    input_image = np.expand_dims(input_image, axis=0)
    preds = vit_object_detector.predict(input_image)[0]

    (h, w) = (im).shape[0:2]

    class_name = preds[0]

    top_left_x, top_left_y = int(preds[1] * w), int(preds[2] * h)

    bottom_right_x, bottom_right_y = int(preds[3] * w), int(preds[4] * h)

    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    # Create the bounding box
    rect = mat_patches.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax1.add_patch(rect)
    ax1.set_xlabel(
        "Predicted: "
        + str(top_left_x)
        + ", "
        + str(top_left_y)
        + ", "
        + str(bottom_right_x)
        + ", "
        + str(bottom_right_y)
    )

    class_name = y_test[0]

    top_left_x, top_left_y = int(y_test[i][1] * w), int(y_test[i][2] * h)

    bottom_right_x, bottom_right_y = int(y_test[i][3] * w), int(y_test[i][4] * h)

    box_truth = top_left_x, top_left_y, bottom_right_x, bottom_right_y

    mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
    # Create the bounding box
    rect = mat_patches.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax2.add_patch(rect)
    ax2.set_xlabel(
        "Target: "
        + str(top_left_x)
        + ", "
        + str(top_left_y)
        + ", "
        + str(bottom_right_x)
        + ", "
        + str(bottom_right_y)
        + "\n"
        + "IoU"
        + str(bounding_box_intersection_over_union(box_predicted, box_truth))
    )
    i = i + 1

print("mean_iou: " + str(mean_iou / len(x_test[:10])))
plt.show()
