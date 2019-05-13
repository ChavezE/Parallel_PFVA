import matplotlib.pyplot as plt
import os
import numpy as np


def createGraphs(P_model):
    
    # Create the directory if necessary
    directory = "Graphs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(P_model)):
        X = P_model[i][:,0]
        for j in range(1,9):
            Y = P_model[i][:,j]
            
            
            myFig = plt.figure()
            plt.title('Y'+str(j)+' according to F'+str(i+1))
            myPlot = plt.plot(X,Y)
            plt.xlabel("F (normalized values)")
            plt.ylabel("Proabilistic values")
            myFig.savefig("Graphs/Graph_Y"+str(j)+'_F'+str(i+1)+'.pdf')
            plt.close(myFig)