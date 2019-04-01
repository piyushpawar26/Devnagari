# imports
import numpy as np
from tqdm import tqdm
import os
import cv2

# constants
DIR = 'static/dataset1/nhcd'
TRAINING_SIZE = 0.8

# order -> consonants, vowels, numerals
# creating labels for the corresponding input
def imageLabel(main_folder, sub_folder):
    one_hot_array_consonants = [0 for _ in range(36)]
    one_hot_array_vowels = [0 for _ in range(12)]
    one_hot_array_numerals = [0 for _ in range(10)]
    if main_folder == 'consonants':
        one_hot_array_consonants[int(sub_folder)-1] = 1
    elif main_folder == 'vowels':
        one_hot_array_vowels[int(sub_folder)-1] = 1
    else:
        one_hot_array_numerals[int(sub_folder)] = 1
    return np.array(one_hot_array_consonants + one_hot_array_vowels + one_hot_array_numerals)

# images are 36*36 -> resizing it to 28*28
def processImage(image_path):
    image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    return np.resize(image[4:33, 4:33], (IMAGE_SIZE, IMAGE_SIZE))

# iterate through each image and creating data
def createData():
    all_data = []
    for main_folder in tqdm(os.listdir(DIR)):
        dir = []
        main_folder_path = os.path.join(DIR, main_folder)
        for sub_folder in tqdm(os.listdir(main_folder_path)):
            sub_dir = []
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
            for image in tqdm(os.listdir(sub_folder_path)):
                image_path = os.path.join(sub_folder_path, image)
                image_label = imageLabel(main_folder, sub_folder)
                image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)) # processImage(image_path)
                sub_dir.append([image, image_label])
            dir.append(sub_dir)
        all_data.append(dir)
    all_data = np.array(all_data)
    np.save('all_data.npy', all_data)
    return all_data

# preprocess the data
def preprocessData(data):
    dummy_1d = np.array([0 for _ in range(58)])
    dummy_2d = np.array([[0 for _ in range(36)] for _ in range(36)])

    X_train = np.array([dummy_2d], dtype=np.int)
    X_test = np.array([dummy_2d], dtype=np.int)
    y_train = np.array([dummy_1d], dtype=np.int)
    y_test = np.array([dummy_1d], dtype=np.int)

    for category in data:
        for sub_categoty in category:
            limit = int(len(sub_categoty) * TRAINING_SIZE)
            X_all_images = np.array([dummy_2d], dtype=np.int)
            y_all_images = np.array([dummy_1d], dtype=np.int)
            for index, image in enumerate(sub_categoty):
                i1 = np.array([image[0]])
                i2 = np.array([image[1]])
                if i1.ndim == 3 and i2.ndim == 2:
                    X_all_images = np.append(X_all_images, np.array([image[0]]), axis=0)
                    y_all_images = np.append(y_all_images, np.array([image[1]]), axis=0)
            X_train = np.append(X_train, X_all_images[1:limit], axis=0)
            y_train = np.append(y_train, y_all_images[1:limit], axis=0)
            X_test = np.append(X_test, X_all_images[limit:], axis=0)
            y_test = np.append(y_test, y_all_images[limit:], axis=0)
    return X_train[1:], y_train[1:], X_test[1:], y_test[1:]


# main function
def main():
    if not os.path.exists('all_data.npy'):
        all_data = createData()
    else:
        all_data = np.load('all_data.npy')
    return preprocessData(all_data)

# ensuring the call from same module
if __name__ == '__main__':
    main()
