import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


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
    module_selection = ("mobilenet_v2_100_224", 224)  # Choose model + input image size
    handle_base, pixels = module_selection
    MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(
        handle_base
    )  # Used to donwload the model
    IMAGE_SIZE = (pixels, pixels)  # image size
    print("[LOG] Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
    BATCH_SIZE = 32  # For training

    print("[LOG] Prepping data")
    fnames, labels = read_image_info()  # Get filnames + labels
    X = np.array(
        [np.array(load_img(f, target_size=IMAGE_SIZE)) / 255.0 for f in fnames]
    )  # load and preprocess images
    lb = LabelBinarizer()
    y = lb.fit_transform(labels)  # Convert labels to one hot encoded labels
    # TODO: split train_test, but for now this is sort of happening in validation_split?

    print("[LOG] Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(MODULE_HANDLE, trainable=False),  # Pretrained model
            tf.keras.layers.Dropout(rate=0.2),  # Training parameter
            tf.keras.layers.Dense(
                10,  # TODO: Set to number of classes automatically
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            ),
        ]
    )
    model.build((None,) + IMAGE_SIZE + (3,))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print("[LOG] Start Training")
    # TODO: add checkpoints
    hist = model.fit(
        X, y, validation_split=0.15, batch_size=BATCH_SIZE, epochs=20, verbose=1
    )
    print("[LOG] Plotting metrics")
    output_training_metrics(hist.history)
    print("[LOG] Saving model")
    model.save(model, "WhaleWhaleWhale")
