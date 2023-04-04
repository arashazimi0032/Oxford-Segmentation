import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
import numpy as np
import os


def load_input_image(path, img_size):
    return img_to_array(load_img(path, target_size=img_size))


def load_target(path, img_size):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img


def get_data_path_list(input_dir, target_dir):
    input_img_paths = sorted([os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith(".jpg")])

    target_paths = sorted([os.path.join(target_dir, name) for name in os.listdir(target_dir) if name.endswith(".png")
                           and not name.startswith(".")])

    return input_img_paths, target_paths


def stack_data(input_img_paths, target_paths, img_size, num_img):
    input_img = np.zeros((num_img,) + img_size + (3,), dtype="float32")
    targets = np.zeros((num_img,) + img_size + (1,), dtype="uint8")

    for i in range(num_img):
        input_img[i] = load_input_image(input_img_paths[i], img_size)
        targets[i] = load_target(target_paths[i], img_size)

    return input_img, targets


def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)


def create_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    return mask


def mat2gray(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255


def stack_mask(pred, image):
    mask = create_mask(np.expand_dims(pred, axis=0))[0]
    pred[:, :, 0] = mat2gray(pred[:, :, 0])
    pred[:, :, 1] = mat2gray(pred[:, :, 1])
    pred[:, :, 2] = mat2gray(pred[:, :, 2])
    out = pred * 0.5 + image * 0.5
    out_r = out[:, :, 0]
    out_g = out[:, :, 1]
    out_b = out[:, :, 2]
    out_r[mask == 127] = image[:, :, 0][mask == 127]
    out_g[mask == 127] = image[:, :, 1][mask == 127]
    out_b[mask == 127] = image[:, :, 2][mask == 127]
    return np.dstack([out_r, out_g, out_b]).astype(np.uint8)
