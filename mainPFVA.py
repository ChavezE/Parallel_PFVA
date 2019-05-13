# ===== LIBRARIES ===== #
import numpy as np
# import SVD
# 

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
    # Read input data from CSV file

    # Center the input ==> i.e. subtract the mean from the data

    # Compute the SINGULAR VALUE DECOMPOSITION

    # Compute the F matrix & Extract the Freduced matrix

    # Create the matrices to fill the P_model list

    # Normalize and obtaint the MIN,MAX from each 
    
    # Final Regression

    # PREDICT!!!
    pass




if __name__ == "__main__":
