# GBH

## Install

Clone and install this image_augmentation package:

```sh
git clone https://github.com/timdeklijn/image_augmentation
cd image_augmentation/im_augment && pip install . && cd ../..
pip install -r requirements.txt
```

## Data preprocessing

Make sure the trainins folder is in the source folder and called `train`. Then run:

```sh
python start.py --dataprep
```

- [x] Get all filenames in the whale folders
- [x] Per whale, create a list of image file names.
- [x] Depending on the length of the list, create augmented images untill each whale has a minimum number of images associated with it.
- [x] Save this augmented images in a separate folder.
- [x] Save a list with all images per whale

## Start training

```sh
python start.py --train
```

- (re) Training code is based on [this](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb#scrollTo=QzW4oNRjILaq)
- [x] Load train set:
    - [x] Based on whale image list, simply load the images and their associated label
- [x] Download a deep learning model
- [x] Retrain a deep learning image classification model
- [x] The output of the model should be a list of probabilities for each whale ID.
- [ ] Runs on GPU
- [ ] Add checkpoints
- [ ] Better train/test split
- [ ] Save label binarizer for inference

## Create output

```sh
python start.py --predict
```

- [ ] The output is a list of test image names with a list of train image names of the same whale.
    - [ ] Run inference on the test images
    - [ ] Per image get a vector of probabilities per whale
    - [ ] per image return file names of images belonging to a certain whale
    - [ ] Put file names with highest probabilities on position 1,2,3 and 20
- [ ] Convert output to whale ID + probability

## TODO

- [ ] Clean up code
- [ ] Add inference

## Stretch

- [ ] tensorboard
- [ ] MLFlow
- [ ] Increase number of images even more