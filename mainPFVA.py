# ===== LIBRARIES ===== #
import numpy as np
from inputProcessor import *
from SVD_Probaility_Tables import *
from linearRegression import *
import argparse
from graphObtention import *


# ===================== #
THETA_FILE = 'thetas.csv'

# ===================== #
'''
Description:
    Utility funciton to create a new model based on images
'''
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
    
    createGraphs(P_model)
    

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