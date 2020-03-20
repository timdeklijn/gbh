import im_augment as aug
from PIL import Image as pil_image
import os
import shutil
import random
import cv2

WHALE_NUM = 5  # What is the minimum number of images per whale?


class WhaleData:
    def __init__(self, id, flist):
        """Simple container class to stare info on the whale training
        data

        Parameters
        ----------
        id : int
            Whale ID
        flist : List(str)
            filenames of images belonging to this ID
        """
        self.id = id
        self.flist = flist
        self.n = len(self.flist)

    def __repr__(self):
        """Modify the print function of this class

        Returns
        -------
        str
            Print string
        """
        return f"ID: {self.id}, num: {self.n}"

    def output_line(self):
        """When writing the whale data, write this line

        Returns
        -------
        str
            String with whale ID and all images
        """
        return f"{self.id},{','.join(self.flist)}"

    def add_image(self, im):
        """Add image to flist, then inrement self.n

        Parameters
        ----------
        im : str
            filename
        """
        self.flist.append(im)
        self.n = len(self.flist)


def handle_input():
    """Go through the folder 'train/' and get all image file names.
    Use the dirpath to get the whale ID. Place this information in a
    WhaleData object and add to a list.

    Returns
    -------
    list(WhaleData)
        Contains information on the training data
    """
    whale_list = []
    ignore_folders = [
        "cf_of_fingers",
        "cf_of_whitehead",
        "nike",
        "",
        "-1",
    ]
    for (dirpath, dirnames, filenames) in os.walk("train/"):
        whale_name = dirpath.replace("train/", "")
        if whale_name not in ignore_folders:
            tmp = []
            for f in filenames:
                tmp.append(os.path.join("train", whale_name, f))
            whale_list.append(WhaleData(int(whale_name), tmp))
    return whale_list


def preprocess_images():
    """Go through all training data and get image names + whale ID's.
    Then, if there are less then WHALE_NUM images, augment the existing ones
    to create extra data. These new images are saved into the folder 'augment'.
    Then save a vsc file with whale id and all image filenames on a line. This
    file is saved as 'train_augment.csv'.
    """
    whale_list = handle_input()
    # Remove augment folder if it exists
    if os.path.exists("augment"):
        shutil.rmtree("augment")
    os.makedirs("augment")  # Create new augmentation folder
    # For all whales check if there is enough images, if not create extra
    for w in whale_list[:40]:  # TODO: remove 10
        tmp = []
        n = 0
        # If there is not enough images
        while w.n + len(tmp) < WHALE_NUM:
            img = pil_image.open(random.choice(w.flist))  # Open a random existing image
            img = aug.random_augment(img)  # Randomly augment this image
            im_name = os.path.join("augment", f"{w.id}_{n}.jpg")  # Create new filename
            aug.save_image(img, im_name)  # Save the image
            tmp.append(im_name)  # Append file name
            n += 1
        # Add new images to WhaleData object
        for n in tmp:
            w.add_image(n)
    # Write existing + augmented images per whale to file
    with open("train_augment.csv", "w") as f:
        for w in whale_list[:40]:
            f.write(w.output_line() + "\n")


## Lorrie: potential start for cropping and masking images:

# def crop(PATH):
    # '''
    # Takes image and makes them greyscale, scales the size of each image to be
    # uniform with the largest image by padding with black pixels, crops
    # image down to minimize ocean background and make whale tale dominate
    # image space.

    # Input: file path with file name
    # Output: cropped and edited image
    # '''

#     h_factor = 2 #here is for 1:"2"

#     img = cv2.imread(PATH,0)

#     v = img.shape[0] # vertial pixels
#     h = img.shape[1] # horizontal pixels

#     #find edges with Canny algorithm
#     canny_edges = cv2.Canny(img,300,300)

#     if v < h/h_factor:
#         fill_length = int(abs(h/h_factor-v)*0.5)#np.random.rand()) # for upper filling

#         # black rectangle for upper filling
#         fill = np.zeros(fill_length* h).reshape(fill_length, h)

#         canny_edges = np.r_[fill,canny_edges,fill] # fill with black rectangle
#         img = np.r_[fill,img,fill] # fill with black rectangle

#     ver = int(h/h_factor)
#     cnt = []
#     for i in range(canny_edges.shape[0]-ver+2):
#         cnt.append(canny_edges[i:i+ver,:].sum()/255)

#     cnt_arr = np.array(cnt)
#     i = cnt_arr.argmax()
#     return PIL.Image.fromarray(np.uint8(img[i:i+ver,:]))
