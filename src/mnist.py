import numpy as np
from matplotlib import pyplot as plt

def load_dataset():
    print("Loading dataset...")
    train_imgs, train_labels = load_file("../res/mnist/mnist_train.csv")
    test_imgs, test_labels = load_file("../res/mnist/mnist_test.csv")
    print("Dataset loaded.")
    return train_imgs, train_labels, test_imgs, test_labels

def load_file(file):
    with open(file, "r") as f:
        loaded = f.readlines()
    imgs = []
    labels = []
    for line in loaded:
        splitted = map(int, line.split(","))
        label, *img = splitted
        imgs.append(img)
        labels.append(label)
    return np.array(imgs) / 255 * 0.99 + 0.01, np.array(labels)

def draw(img):
    arr2d = 1 - np.reshape(img, (28, 28))
    plt.imshow(arr2d, cmap='gray')