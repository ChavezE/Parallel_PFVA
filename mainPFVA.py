# ===== LIBRARIES ===== #
import numpy as np
# from SVD import *
from inputProcessor import *
from SVD_Probaility_Tables import *
from linearRegression import *

# ===================== #

# ===== NORMALIZATION FUNCTIONS ===== #
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
    normColVector = origColVector / maxVal
    minVal = normColVector[0] 

    if verbose:
        print("normalized vector : {}".format(normColVector))

    ansVals = {
        "normColVector" : normColVector,
        "MAX" : 1.,
        "MIN" : minVal
    }

    return ansVals

# =================================== #

def main():
    # Read input data from CSV file TODO Emilio
    Xorig, Yorig = retrieveInputDataXMatrix(CSV_FILE_NAME), retrieveInputDataYMatrix(CLASS_DESCRIPTION_FILE)

    # Center the input ==> i.e. subtract the mean from the data
    Xc = centerData(Xorig)

    # Compute the F matrix & Compute the SINGULAR VALUE DECOMPOSITION
    F = compute_F_Mat(Xc)

    # Extract the Freduced matrix
    K = 5 # Number of F columns to be considered
    Freduced = F[:,:K]

    # Create the matrices to fill the P_model list
    P_model = []
    for i in range(K):
        probability_Matrix = probability_Estimate(Freduced[:,i],Yorig)
        P_model.append(probability_Matrix)
    print(P_model)

    # Normalize and obtaint the MIN,MAX from each 
    for i in range(K):
        currentProbMat = P_model[i]
        normDict = normalizeFthRow(currentProbMat)
        currentProbMat[:,0] = normDict["normColVector"]

    # Final Regression TODO Emilio

    # PREDICT!!!
    




if __name__ == "__main__":
    main()