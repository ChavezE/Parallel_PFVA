import numpy as np

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





################################################################################
############################## DEPRECIATED ######################################
################################################################################

'''
Description:
    This function computes vector response Y from the estimators obtained
    from the predictive model stage. This vector is an average of each
    estimator k, for each one of the estimated classes, in this case 8.
    INPUT:
        P_model: list of numpy vectors
'''
def computeResponseVectorYAverage(P_model):
    assert len(P_model) > 1
    rows, cols = P_model[0].shape
    print(rows,cols)
    # Has the average of all stimatiors for each Yi
    Y_vector_repsonse = np.zeros(shape=(rows,cols-1)) # remove fn

    # iterate the list and take the average
    for subMat in P_model:
        # for each k estimator, for our proj always 8
        for row in range (rows):
            for k in range(1,cols): # removing F
                # sum each k row
                Y_vector_repsonse[row,k-1] = Y_vector_repsonse[row,k-1] + subMat[row,k]
    
    for r in range(rows):
        t_row = Y_vector_repsonse[r,:]
        t_row = t_row / cols
        Y_vector_repsonse[r,:] = t_row

    return Y_vector_repsonse

if __name__ == '__main__':
    P_model = []
    P_model.append(np.asarray([ [1,1,3], 
                                [5,6,7]])) # mocking F1

    P_model.append(np.asarray([ [5,2,3], 
                                [5,6,7]])) # mocking F1

    P_model.append(np.asarray([ [5,3,3], 
                                [5,6,7]])) # mocking F1
    print(P_model)
    print (computeResponseVectorYAverage(P_model))

    