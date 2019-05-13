import numpy as np
from SVD_Probaility_Tables import *
import argparse
import sys
# Temporary workaround for fixing ROS problem in Emilio's machine
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

THETA_FILE = 'thetas.csv'


def test_image(img_name):
    cur_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # print (cur_img.shape)
    img_row = np.reshape(cur_img, (1, 256*256))
    Xc = centerData(img_row)
    print(Xc[0,40000:40500])

    # Compute the F matrix & Compute the SINGULAR VALUE DECOMPOSITION
    print("\n===== Computing F matrix through SVD =====\n")    
    F = compute_F_Mat(Xc)


    # Extract the Freduced matrix
    K = 5 # Number of F columns to be considered
    Freduced = F[:,:K]

    # load theta vector
    thetas = np.genfromtxt(THETA_FILE, delimiter=',')

    # Predict
    print ("F shape:", F.shape)
    print ("F Reduced shape:", Freduced.shape)
    print (Freduced)
    print ("Thetha shape:", thetas.shape)
    resultsMatrix = Freduced @ thetas

    print("\n===== Dimensions of the Result Matrix {} =====\n".format(resultsMatrix.shape))
    print (resultsMatrix) 


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_location", required=True,help="Path to the image we want to test")
    args = vars(ap.parse_args())

    test_image(args["img_location"])
    