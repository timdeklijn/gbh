# import os
import math

import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import keras


def create_VGG_model():
    model = VGG19(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = keras.layers.Flatten()(model.output)
    class1 = keras.layers.Dense(128, activation="sigmoid")(flat1)
    out = keras.layers.Dense(
        40,  # TODO: Set to number of classes automatically
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    )(class1)
    model = keras.models.Model(inputs=model.inputs, outputs=out)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model


class Sequencer(keras.utils.Sequence):
    def __init__(self, X_train, y_train, batch_size, image_size):
        self.x, self.y = X_train, y_train
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_train = []
        y_train = []
        for i in range(len(batch)):
            X_train.append(
                np.array(load_img(self.x[i], target_size=self.image_size)) / 255.0
            )
            y_train.append(self.y[i])
        return np.array(X_train), np.array(y_train)


def read_image_info():
    """Read train_augment file which contains all images per whale,
    both original and augmented. Then split this up in a list of
    filenames and a list of labels

    Returns
    -------
    files, labels
        List of files and corresponding list of labels
    """
    labels = []
    files = []
    with open("train_augment.csv", "r") as f:
        lines = f.readlines()
    for l in lines:
        spl = l.strip().split(",")
        label = spl[0]
        for f in spl[1:]:
            files.append(f)
            labels.append(label)
    return files, labels


def output_training_metrics(hist):
    """Plot accuracy and loss metrics from hist dict

    Parameters
    ----------
    hist : dict
        Contains training metrics
    """
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("train_acc.png")
    plt.clf()

    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("train_loss.png")


def train_model():

    # Model parameters
    IMAGE_SIZE = (224, 224)  # image size
    BATCH_SIZE = 25  # For training

    print("[LOG] Prepping data")
    fnames, labels = read_image_info()  # Get filnames + labels
    print("[LOG] Fixing labels")
    lb = LabelBinarizer()  # Save this!!!
    y = lb.fit_transform(labels)  # Convert labels to one hot encoded labels
    print("[LOG] Creating generators")
    X_train, X_test, y_train, y_test = train_test_split(fnames, y, test_size=0.15)
    num_train, num_val = len(X_train), len(X_test)

    train_generator = Sequencer(X_train, y_train, BATCH_SIZE, IMAGE_SIZE)
    val_generator = Sequencer(X_test, y_test, BATCH_SIZE, IMAGE_SIZE)

    # TODO: Make sure that only original images are put into the test set? so maybe do
    #       this before aumenting the images then.
    # TODO: save the label binarizer to get back the whale ID after predicitons

    print("[LOG] Building model with")

    # TODO: add checkpoints, save every N epochs.
    #
    # From Monica:
    # EPOCH_BLOCK = 10
    # N_EPOCH_BLOCKS = 10
    # for i in range(N_EPOCH_BLOCKS):
    #     epoch_start = i * EPOCH_BLOCK
    #     hist = model.fit(
    #         X, y, validation_split=0.15, batch_size=BATCH_SIZE,
    #               epochs=epoch_start + EPOCH_BLOCK, verbose=1,
    #               initial_epoch=epoch_start
    #     )
    # model.save_weights("weights")

    model = create_VGG_model()
    print("[LOG] Start Training")
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // BATCH_SIZE,
        epochs=5,
        validation_data=val_generator,
        validation_steps=num_val // BATCH_SIZE)

    print("[LOG] Plotting metrics")
    output_training_metrics(hist.history)

    print("[LOG] Saving model")
    keras.models.save_model(model, "WhaleWhaleWhale")
