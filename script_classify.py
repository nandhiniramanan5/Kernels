import csv
import random
import numpy as np

import algorithms as algs
import utilities as utils
 
def splitdataset(dataset, trainsize=300, testsize=100):
    # Now randomly split into train and test    
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    offset = 50 # Ignore the first 50 features
    Xtrain = dataset[randindices[0:trainsize],offset:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],offset:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))

def Gaussian(x,z,sigma):
    #gaussian kernel funtion
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
    
def sigmoid(x,y):
    #sigmoid kernel function
    alpha=0.001
    return np.tanh(alpha*np.dot(x.T,y))
    
def polynomial(slope,x,y,c,d):
    #polynomial kernel function
    return ((slope*np.dot(x.T,y)+c)**d)
 
def geterror(predictions, ytest):
    # Can change this to other error values
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]
 
if __name__ == '__main__':
    slope=.00000000000001
    d=2
    c=1
    filename = 'blogData_train.csv'
    dataset = utils.loadcsv(filename)   
    trainset, testset = splitdataset(dataset)
    exp = [x for x in range(1,150)]
    Xtrain=trainset[0]
    Xtest=testset[0]
    Center=Xtrain[exp,:]
    param=np.var(Center)
    #ktrain and Ktest initialisation
    Ktrain=np.zeros(shape=(Xtrain.shape[0],Center.shape[0]))
    Ktest=np.zeros(shape=(Xtest.shape[0],Center.shape[0]))
    n=1
    while n==1:
        x=input("enter the type of kernel you wish to use 1.linear 2. polynomial 3.guassian 4.Sigmoid  5.Without kernel: ")
        if x==1:
            #linear kernel
            Ktrain=np.dot(trainset[0],Center.T)
            Ktest=np.dot(testset[0],Center.T)
            
        elif x==2:
            #polynomial kernel
            i=0
            for v_i in Xtrain:
                j=0
                for v_j in Center:
                     Ktrain[i,j]=polynomial(slope,v_i,v_j.T,c,d)                    
                     j+=1
                i+=1
            i=0    
            for v_i in Xtest:
                j=0
                for v_j in Center:
                    Ktest[i,j]=polynomial(slope,v_i,v_j.T,c,d)
                    j+=1
                i+=1        
        elif x==3:
            #Gaussian kernel
            i=0
            for v_i in Xtrain:
                j=0
                for v_j in Center:
                    Ktrain[i,j]=Gaussian(v_i,v_j.T,param)
                    j+=1
                i+=1
            i=0
            for v_i in Xtest:
                j=0
                for v_j in Center:
                    Ktest[i,j]=Gaussian(v_i,v_j.T,param)
                    j+=1
                i+=1
        elif x==4:
            #Sigmoid kernel
            i=0
            for v_i in Xtrain:
                j=0
                for v_j in Center:
                    Ktrain[i,j]=sigmoid(v_i,v_j.T)
                    j+=1
                i+=1
            i=0
            for v_i in Xtest:
                j=0
                for v_j in Center:
                    Ktest[i,j]=sigmoid(v_i,v_j.T)
                    j+=1
                i+=1
        elif x==5:
            Ktrain=trainset[0]
            Ktest=testset[0]
    
        print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), trainset[0].shape[0], testset[0].shape[0])
        classalgs = {#'Random': algs.Regressor(),
                 #'Mean': algs.MeanPredictor(),
                 'FSLinearRegression': algs.FSLinearRegression()
                 }

        # Runs all the algorithms on the data and print out results    
        for learnername, learner in classalgs.iteritems():
            print 'Running learner = ' + learnername
            # Train model
            learner.learn(Ktrain, trainset[1])
            # Test model
            predictions = learner.predict(Ktest)
            #print predictions
            accuracy = geterror(testset[1], predictions)
            print 'L2 norm for ' + learnername + ': ' + str(accuracy)
        n=input("enter 1 if you wish to continue: ")
 
