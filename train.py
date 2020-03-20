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


# def generator(samples, image_size, batch_size=32, resize=224):
#     """
#     Yields the next training batch.
#     Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],.]
#     """
#     num_samples = len(samples)
#     while True:
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset: offset + batch_size]
#             X_train = []
#             y_train = []
#             for batch_sample in batch_samples:
#                 X_train.append(
#                     np.array(load_img(batch_sample[0], target_size=image_size))
#                 )
#                 y_train.append(batch_sample[1])
#             yield np.array(X_train), np.array(y_train)


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
    # module_selection = ("mobilenet_v2_100_224", 224)  # Choose model + input image size
    # handle_base, pixels = module_selection
    # MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(
    #     handle_base
    # )  # Used to donwload the model
    IMAGE_SIZE = (224, 224)  # image size
    # print("[LOG] Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
    BATCH_SIZE = 25  # For training

    print("[LOG] Prepping data")
    fnames, labels = read_image_info()  # Get filnames + labels
    print(len(fnames), "YOOOLOOOOOO==================================================")
    print("[LOG] Fixing labels")
    lb = LabelBinarizer()  # Save this!!!
    y = lb.fit_transform(labels)  # Convert labels to one hot encoded labels
    print("[LOG] Creating generators")
    X_train, X_test, y_train, y_test = train_test_split(fnames, y, test_size=0.15)
    num_train, num_val = len(X_train), len(X_test)

    train_generator = Sequencer(X_train, y_train, BATCH_SIZE, IMAGE_SIZE)
    val_generator = Sequencer(X_test, y_test, BATCH_SIZE, IMAGE_SIZE)

    # train_generator = generator(
    #     train_list, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    # )
    # val_generator = generator(y, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    # TODO: Make sure that only original images are put into the test set? so maybe do
    #       this before aumenting the images then.
    # TODO: save the label binarizer to get back the whale ID after predicitons

    print("[LOG] Building model with")

    # model = tf.keras.Sequential(
    #     [
    #         hub.KerasLayer(MODULE_HANDLE, trainable=False),  # Pretrained model
    #         tf.keras.layers.Dropout(rate=0.2),  # Training parameter
    #         tf.keras.layers.Dense(
    #             10,  # TODO: Set to number of classes automatically
    #             activation="softmax",
    #             kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    #         ),
    #     ]
    # )
    # model.build((None,) + IMAGE_SIZE + (3,))
    # model.summary()
    # model.compile(
    #     optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    #     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    #     metrics=["accuracy"],
    # )
    # TODO: add checkpoints, save every N epochs.
    #
    # From Monica:
    # EPOCH_BLOCK = 10
    # N_EPOCH_BLOCKS = 10
    # for i in range(N_EPOCH_BLOCKS):
    #     epoch_start = i * EPOCH_BLOCK
    #     hist = model.fit(
    #         X, y, validation_split=0.15, batch_size=BATCH_SIZE, epochs=epoch_start + EPOCH_BLOCK, verbose=1,
    #         initial_epoch=epoch_start
    #     )
    # model.save_weights("weights")

    # TODO: add generator, we can not have all images in memory
    # Dang-Khoa:
    #   https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-2-be9ad08f3f0e
    #

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
    keras.models.save_model("WhaleWhaleWhale")


# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 224, 224, 3)       0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 25088)             0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               3211392   
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 23,237,066
# Trainable params: 3,212,682
# Non-trainable params: 20,024,384
# _________________________________________________________________

# [LOG] Start Training
# Epoch 1/5
# 2020-03-20 15:23:28.028786: W tensorflow/core/common_runtime/base_collective_executor.cc:217] BaseCollectiveExecutor::StartAbort Invalid argument: Matrix size-incompatible: In[0]: [1,526848], In[1]: [25088,128]
#          [[{{node dense_1/Relu}}]]
# Traceback (most recent call last):
#   File "start.py", line 42, in <module>
#     train_model()
#   File "/home/tim/projects/gbh/train.py", line 207, in train_model
#     validation_steps=num_val // BATCH_SIZE)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
#     return func(*args, **kwargs)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/keras/engine/training.py", line 1732, in fit_generator
#     initial_epoch=initial_epoch)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/keras/engine/training_generator.py", line 220, in fit_generator
#     reset_metrics=False)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/keras/engine/training.py", line 1514, in train_on_batch
#     outputs = self.train_function(ins)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/keras/backend.py", line 3727, in __call__
#     outputs = self._graph_fn(*converted_inputs)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1551, in __call__
#     return self._call_impl(args, kwargs)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1591, in _call_impl
#     return self._call_flat(args, self.captured_inputs, cancellation_manager)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1692, in _call_flat
#     ctx, args, cancellation_manager=cancellation_manager))
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 545, in call
#     ctx=ctx)
#   File "/home/tim/projects/gbh/venv/lib/python3.6/site-packages/tensorflow_core/python/eager/execute.py", line 67, in quick_execute
#     six.raise_from(core._status_to_exception(e.code, message), None)
#   File "<string>", line 3, in raise_from
# tensorflow.python.framework.errors_impl.InvalidArgumentError:  Matrix size-incompatible: In[0]: [1,526848], In[1]: [25088,128]
#          [[node dense_1/Relu (defined at /home/tim/projects/gbh/venv/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_1069]

# Function call stack:
# keras_scratch_graph