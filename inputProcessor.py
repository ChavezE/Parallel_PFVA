import numpy as np 
import sys
import os
import glob 
# Temporary workaround for fixing ROS problem in Emilio's machine
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

# ----------------------- Constants ----------------------- #
IMAGES_FOLDER = 'images/'
RESIZED_IMAGES_FOLDER = 'resized_images/'
CSV_FILE_NAME = 'images_csv_data.csv'

# ----------------------- Functions ----------------------- #

'''
Description:
    Resises images in images_directory and stores them in new_dir_name
    images are resiezed to 256x256
'''
def resizeImagesTo256(images_directory, new_dir_name):
    # create new dir 
    os.mkdir(new_dir_name)
    
    for img in images_directory:
        print ("Reading... {}".format(img))
        # load image
        cur_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # resize image
        newimg = cv2.resize(cur_img,(int(256),int(256)))
        # rename and write
        newimage_name_with_extention = str(img[-10:])

        # change image to jpg since opencv cant write .pgm images
        new_image = newimage_name_with_extention[:-3] + 'jpg' 
        print("new name", new_image)
        print ("Writing...{} resized to 256x256".format(RESIZED_IMAGES_FOLDER + new_image))
        cv2.imwrite(RESIZED_IMAGES_FOLDER + new_image, newimg)

'''
Description:
    Creates a Numpy matrix storing each pixel value per row.
    a CSV file is created to store the matrix created name given by file_name
'''
def createInputDataXMatrix(images_directory, file_name):
    # list the folder to iterate it
    images_list = sorted(glob.glob(images_directory + "*.jpg"))
    # create numpy matrix
    input_datax_matrix = np.zeros(shape=(322, 256*256), dtype=np.uint8)
    i = 0
    for img in images_list:
        print (img)
        # load image
        cur_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # print (cur_img.shape)
        img_row = np.reshape(cur_img, (1, 256*256))
        # print (img_row.shape)
        # print("cur img MAX : {}, MIN: {}".format(img_row.max(), img_row.min()))
        input_datax_matrix[i] = img_row
        i = i + 1
    
    # create a csv of the matrix created
    np.savetxt(file_name, input_datax_matrix, delimiter=",", fmt="%3u")
    print ("created file: {}".format(file_name))

'''
Description:
    Parses CSV and returns numpy mat shape=(322, 256*256)
'''
def retrieveInputDataXMatrix(file_name):
    # create numpy matrix
    input_datax_matrix = np.zeros(shape=(322, 256*256), dtype=np.uint8)
    input_datax_matrix = np.genfromtxt(file_name, delimiter=',')

    return input_datax_matrix


# ----------------------- Main ----------------------- #

if __name__ == '__main__':
    images_directory = sorted(glob.glob(IMAGES_FOLDER + "*.pgm"))

    # call resizer
    # resizeImagesTo256(images_directory, RESIZED_IMAGES_FOLDER)
    
    # call input_datax creator
    # createInputDataXMatrix(RESIZED_IMAGES_FOLDER, CSV_FILE_NAME)
    retrieveInputDataXMatrix(CSV_FILE_NAME)