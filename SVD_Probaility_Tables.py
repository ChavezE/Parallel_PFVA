import numpy as np
import pandas as pd

def centerData(X):
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    return X_c

def compute_F_Mat(X_c):
    '''
    Compute the Singular Value Decomposition & F matrix

    Return Values
        - F matrix (Fv1)
    '''
    Rows, Cols = X_c.shape
    Cols = Rows

    P, Dvec, Q_t = np.linalg.svd(X_c, full_matrices=False, compute_uv=True)
    Q = Q_t.T
    D = np.zeros((Rows, Cols))
    D[:Cols,:Cols] = np.diag(Dvec)
    Fv1 = P @ D
    Fv2 = X_c @ Q
    assert(Fv1.all() == Fv2.all())
    return Fv1

# Recives a column of F and the matrix Y, returns the prob matrix for that F value
def probability_Estimate(F,Y):
    
    rows = F.shape
    
    #print("Matrix F for probability : ",rows)
    n = 11   # Set by the profesor
    
    row_Prob_Matrix = rows[0] - int(n/2) - int(n/2)
    col_Prob_Matrix = 8

    # se crea la matriz de probabilidad
    prob_Matrix = np.zeros((row_Prob_Matrix,col_Prob_Matrix+1))

    base_Matrix = np.zeros((rows[0],col_Prob_Matrix+1))
    base_Matrix[:,0] = F
    base_Matrix[:,1:] = Y
    
    #print("Base matrix",base_Matrix)

    # Sort the matrix according to F values
    base_Matrix_Sorted = base_Matrix[base_Matrix[:,0].argsort()]
    #print("Base matrix ordenada",base_Matrix_Sorted)

    for i in range (row_Prob_Matrix):
        for j in range(8):
            prob_Matrix[i,j+1] = np.average(base_Matrix_Sorted[i:i+n,j+1])
    rowBase,colBase = base_Matrix_Sorted.shape
    prob_Matrix[:,0] = base_Matrix_Sorted[int(n/2):rowBase - int(n/2),0]

    
    
    return prob_Matrix




def main():
    # ======= Data set 1 ============
    X_input = pd.read_excel("Clean Data.xlsx")
    X = X_input.values
    X = X[:,2:22]
    Rows, Cols = X.shape
    print("Size of X",X.shape)

    Y_input = pd.read_excel("classes_example.xlsx")

    # Amounts of F's that are desired
    K = 5

    ############################## Erase this for real test#######################3333
    Y = np.zeros((87,8))
    Y[:,0] = Y_input.values.T
    Y[:,1] = Y_input.values.T
    Y[:,2] = Y_input.values.T
    Y[:,3] = Y_input.values.T
    Y[:,4] = Y_input.values.T
    Y[:,5] = Y_input.values.T
    Y[:,6] = Y_input.values.T
    Y[:,7] = Y_input.values.T

    print("Shape of Y",Y.shape)
    ##########################################################

    ####################### Uncomment this for real test####################
    #Y = Y_input
    ######################################################################

    X_c = centerData(X)
    F = compute_F_Mat(X_c)
    
    P_model = []

    for i in range(K):
        probability_Matrix = probability_Estimate(F[:,i],Y)
        P_model.append(probability_Matrix)
    

    
    
    
    print("==============F===========")
    print(F)
    print("==============Probability===========")
    print(P_model)

if __name__ == "__main__":
    main()