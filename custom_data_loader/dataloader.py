from __future__ import division  # floating point division
import math
import numpy as np

####### Main load functions
def load_ocean(train_filename, test_filename):
    """ A ocean dataset """
    filename = 'datasets/ocean.csv'
    train_dataset = loadcsv(train_filename)
    test_dataset = loadcsv(test_filename)
    trainset, testset = splitdataset(train_dataset, len(train_dataset), 0, output_len=1, testdataset=test_dataset, featureoffset=50)    
    return trainset,testset


####### Helper functions

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, output_len=1, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    featureend = dataset.shape[1]-output_len
    outputlocation = featureend    
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0
    
    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation:]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation:]

    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation:]        

    """
    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility    
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)
                        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    """
    return ((Xtrain,ytrain), (Xtest,ytest))