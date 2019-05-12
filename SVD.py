import numpy as np
import pandas as pd

def centerData(X):
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    return X_c

def compute_F_Mat(X_c):
    Rows, Cols = X_c.shape

    # a[a[:,0].argsort()]

    P, Dvec, Q_t = np.linalg.svd(X_c, full_matrices=True, compute_uv=True)
    Q = Q_t.T
    D = np.zeros((Rows, Cols))
    D[:Cols,:Cols] = np.diag(Dvec)
    Fv1 = P @ D
    Fv2 = X_c @ Q
    assert(Fv1.all() == Fv2.all())
    return Fv1

def probability_Estimate(F,Y):
    rows,cols = F.shape
    print(rows,cols)

    n =11   # Set by the profesor
    row_Prob_Matrix = rows - int(n/2)
    col_Prob_Matrix = cols
    prob_Matrix = np.zeros((row_Prob_Matrix,col_Prob_Matrix))
    print(prob_Matrix.shape)
    for j in range(col_Prob_Matrix):
        for i in range(row_Prob_Matrix):
            prob_Matrix[i,j] = np.average(F[i:i+n,j])
    
    return prob_Matrix




def main():
    # ======= Data set 1 ============
    X_input = pd.read_excel("Clean Data.xlsx")
    X = X_input.values
    X = X[:,2:22]

    Y_input = pd.read_excel("classes_example.xlsx")

    Rows, Cols = X.shape
    print(X.shape)
    # ======= Data set 2 ============
    # X_t = np.array([
    #     [3,6,2,6,2,9,6,5,9,4,7,11,5,4,3,9,10,5,4,10],
    #     [14,7,11,9,9,4,8,11,5,8,2,4,12,9,8,1,4,13,15,6]
    # ])
    # X = X_t.T

    X_c = centerData(X)
    F = compute_F_Mat(X_c)
    probability_Matrix = probability_Estimate(F,Y_input)
    print("==============F===========")
    print(F)
    print("==============Probability===========")
    print(probability_Matrix)

if __name__ == "__main__":
    main()