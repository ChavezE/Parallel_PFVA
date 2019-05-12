import numpy as np
import math
import pandas as pd

def PCA(X):
    ren,col = X.shape
    # print(ren,col)
    Xc = ( X - np.mean(X) ) / ( np.std(X) )
    # print(X)

    # for i in range(col):
    #     X[:,i] = (X[:,i] - X[:,i].mean())/(math.sqrt(ren-1) *X[:,i].std() )
    #     #print(X[:,i].mean())
    #     #print(X[:,i].std())
    A = np.dot(Xc.T , Xc)
    print(A)
    sizeSRen, sizeSCol = A.shape
    S = np.identity(sizeSRen)
    # print(S)

    # Sort Eigen values
    print('\n\n')
    TempA = np.diag(A)
    TempASorted = np.sort(TempA)
    TempASortedReversed = TempASorted[::-1]
    # print(TempA)
    # print(TempASorted)
    # print(TempASortedReversed)

    for i in range(40):
        Q,R = np.linalg.qr(A)
        A = np.dot(R , Q)
        S = np.dot(S , Q)

    print("########################## D\n")
    print (A.shape)
    print(np.diag(A)) # D Eigenvalues
    
    print("########################## Q\n")
    print (S.shape)
    print(S) # Q Eigenvectors

    F = np.dot(Xc , S) 
    print("########################## F\n")
    print (F.shape)
    print(F)

def SVC(X):
    ren,col = X.shape
    # print(ren,col)
    Xc = ( X - np.mean(X) ) / ( np.std(X) )
    # print(X)
    P,D,Q = np.linalg.svd(Xc,full_matrices=True,compute_uv=True)
    print('P',P)
    print('D',D)
    print('Q',Q)


def main():
    X_input = pd.read_excel("Clean Data.xlsx")
    X = X_input.values
    X = X[:,2:22]
    PCA(X)
    print("SVC!!!!!")
    SVC(X)

if __name__ == "__main__":  
    main()