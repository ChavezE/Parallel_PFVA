import numpy as np

'''
Description:
    Implements a curve fitting with the following relationship:
    betha =(Xt*X)^(-1)Xt*y.
    where X is the data set and Y is the class target

    Y = b1*x1 + b2*x2 ... b5*x5
'''
def curveFitting_LSM(X, Y):
    X_inverse = np.linalg.inv(X)
    X_transposed = X.X_transpose()

    # get the fitted coefficients
    betha = np.linalg.inv(X_transposed*X)*X_transposed*Y

    return betha

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

    