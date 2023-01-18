import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.patches as mat_patches
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2

from test_train import Patches

img_path = 'data/test/plane.jpg'

patch_size = 32  # Size of the patches to be extracted from the input images

image_size = 224


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

vit_object_detector = keras.models.load_model('vit_object_detector.h5')

input_image = keras.utils.img_to_array(cv2.imread(img_path))

input_image = input_image.resize((image_size, image_size))

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

top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)

bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

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

top_left_x, top_left_y = int(input_image[0][0] * w), int(input_image[0][1] * h)

bottom_right_x, bottom_right_y = int(input_image[0][2] * w), int(input_image[0][3] * h)

box_truth = top_left_x, top_left_y, bottom_right_x, bottom_right_y

iou = bounding_box_intersection_over_union(box_predicted, box_truth)

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
