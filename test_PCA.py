import numpy as np
import math

# Profesor
def PCA(X):
    ren,col =  X.shape
    print(ren,col)
    '''for i in range(col):
        X[:,i] = (X[:,i] - X[:,i].mean())/(math.sqrt(ren-1) *X[:,i].std() )
        #print(X[:,i].mean())
        #print(X[:,i].std())'''
    A = X.T * X
    print(A)
    sizeSRen,sizeSCol = A.shape
    S = np.identity(sizeSRen)
    print(S)
    for i in range(40):
        Q,R = np.linalg.qr(A)
        A = R*Q
        S = S*Q
    print(A) # D
    print(S) # Q

    F =X*S 
    print(F)


# Web page
def pca(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    #assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    print(eigen_vals)
    print(eigen_vecs)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca

def main():
    X = np.matrix('1 1 ; 0 1 ; -1 1')
    PCA(X)
    print('###############')
    print(pca(X))

if __name__ == "__main__":
    main()