import tensorflow as tf


def predict():

    model = tf.keras.models.load_model("WhaleWhaleWhale")
    model.summary()

    # Example of predict function. This assumes some image preprocessing, see train.py
    # print("tst:", model.predict(X[0].reshape((1,) + X[0].shape)), "label:", y[0])

    # TODO: Load all test images
    # TODO: run prediction on all test images
    # TODO: Load train_augment.csv, a list of whale ID's and image names
    # TODO: based on probability, pick image file names from train_augment.csv, ignore
    #   augmented images.
    # TODO: Write to a csv file like: test_image_filename, img1, img2, ..., img 20
    # TODO: Score seems to be focuessed on img 1,2,3 and 20 so focus on this.
    # TODO: Run local tests first
