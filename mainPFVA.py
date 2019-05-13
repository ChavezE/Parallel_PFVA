# ===== LIBRARIES ===== #
import numpy as np
from inputProcessor import *
from SVD_Probaility_Tables import *
from linearRegression import *
import argparse


# ===================== #
THETA_FILE = 'thetas.csv'

# ===== HELPER FUNCTIONS ===== #
def normalizeFthRow(FProbMat, verbose=False):
    '''
    This function will normalize [0,1) the values from the column 0, i.e. 
    normalize the values corresponding to the F matrix.

    Return Values
        Dictionary with the following elements:
        - normColVector : Normalized column vector
        - MAX : maximun value of the normColVector
        - MIN : minimun value of the normColVector
    '''
    origColVector = FProbMat[:,0]

    if verbose:
        print("original vector : {}".format(origColVector))

    maxVal = origColVector[-1] # The last element is the max value bc preordering
    minVal = origColVector[0] 
    normColVector = origColVector / maxVal

    if verbose:
        print("normalized vector : {}".format(normColVector))

    ansVals = {
        "normColVector" : normColVector,
        "MAX" : maxVal,
        "MIN" : minVal
    }

    return ansVals

# These functions will be used to compute the coefficients theta to predict on new images
'''
Description:
    This function computes matrix Ynorm that will further be used to compute the
    coefficients of the matrix Theta.

    INPUT:
        P_model: list of numpy vectors
'''
def computeYAndFNormalized(P_model):
    assert len(P_model) > 1
    hSamples, cols = P_model[0].shape
    numOfClasses = cols - 1 # Remove fn
    K = len(P_model)

    Ynorm = np.zeros(shape=(hSamples, numOfClasses)) 
    Fnorm = np.zeros(shape=(hSamples, K))

    # iterate the list and take the average
    for i in range(K):
        # Currect Matrix
        subMat = P_model[i]
        
        # Append Fcolumn to Fnorm matix
        Fnorm[:,i] = subMat[:,0]

        # for each k estimator, for our proj always 8
        for row in range(hSamples):
            for k in range(numOfClasses): # removing F
                # sum each k row
                Ynorm[row,k] += subMat[row,k+1]
    
    Ynorm /= (1/K)

    return Ynorm, Fnorm

'''
Description:
    Implements a curve fitting with the following relationship:
    betha =(Xt*X)^(-1)Xt*y.
    where X is the data set and Y is the class target

    Y = b1*x1 + b2*x2 ... b5*x5
'''
def curveFitting_LSM(Yn, Fn):
    # Theta = (Fn' @ Fn)^-1 @ Fn' @ Yn

    theta = np.linalg.inv(Fn.T @ Fn)
    theta = theta @ Fn.T @ Yn

    return theta

def generateModel():
    # Read input data from CSV file 
    print("\n===== Reading input data from CSV file =====\n")
    Xorig, Yorig = retrieveInputDataXMatrix(CSV_FILE_NAME), retrieveInputDataYMatrix(CLASS_DESCRIPTION_FILE)

    # Center the input ==> i.e. subtract the mean from the data
    print("\n===== Centering input data =====\n")
    Xc = centerData(Xorig)

    # Compute the F matrix & Compute the SINGULAR VALUE DECOMPOSITION
    print("\n===== Computing F matrix through SVD =====\n")    
    F = compute_F_Mat(Xc)

    # Extract the Freduced matrix
    K = 5 # Number of F columns to be considered
    Freduced = F[:,:K]

    # Create the matrices to fill the P_model list
    print("\n===== Creating the P Model list of Matrices =====\n")    
    P_model = []
    for i in range(K):
        probability_Matrix = probability_Estimate(Freduced[:,i],Yorig)
        P_model.append(probability_Matrix)
    # print(P_model)

    # Normalize and obtaint the MIN,MAX from each \
    print("\n===== Normalizing P_model =====\n")
    # TODO store data for each index in P_model
    for i in range(K):
        currentProbMat = P_model[i]
        normDict = normalizeFthRow(currentProbMat)
        P_model[i][:,0] = normDict["normColVector"]

    # Final Regression TODO Emilio
    print("\n===== Compute the matrices Ynorm & Fnorm =====\n")
    Ynorm, Fnorm = computeYAndFNormalized(P_model)
    print("\n===== Dimensions of the matrix Ynorm {} =====\n".format(Ynorm.shape))
    print(Ynorm)
    print("\n===== Dimensions of the matrix Fnorm {} =====\n".format(Fnorm.shape))
    print(Fnorm)

    print("\n===== Compute the matrix theta =====\n")
    theta = curveFitting_LSM(Ynorm, Fnorm)
    print("\n===== Dimensions of the matrix theta {} =====\n".format(theta.shape))
    print(theta)

    return theta

def safeThetaCoeff(file_name, thetas):
    np.savetxt(file_name, thetas, delimiter=",", fmt="%1.6f")


# =================================== #
def main(target):
    if target == 'GENERATE':
        print("\n===== Begining model generation =====\n")
        
        # THE JUICE IS CREATED HERE
        thetas = generateModel()
        
        print("\n===== Model generation completed, storing theta values ... =====\n")
        safeThetaCoeff(THETA_FILE, thetas)
        print("\n===== ... Done =====\n")
    
    else:
        # DRINK THE JUICE HERE
        pass



if __name__ == "__main__":
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target", required=True,help="M for generating new model, P for predicting")
    args = vars(ap.parse_args())
   
    # Modeling selected
    if (args["target"] == "M"):
        print ("################################################")
        print ("\n######## Target selected is Modeling ########\n")
        print ("################################################")
        
        target = 'GENERATE'
    
    # Predictor selected
    elif (args["target"] == "P"):
        print ("################################################")
        print ("\n######## Target selected is Prediction ########\n")
        print ("################################################")
        
        target = 'PREDICT'
    
    # pass the desired target to main function
    main(target)