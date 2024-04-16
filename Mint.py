
import pandas as pd
import numpy as np

import pickle


def softmax(x):
    '''
        Compute softmax values for each sets of scores in x.
    '''
    # substract the max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Mint(object):
    '''
    This is a minimized model that is intended for inference only, 
    with on dependency on the Pytorch framework. Once one obtains a model with the 
    `MNL` module, one could *export" the trained model to `Mint` and deploy it in 
    the running time with minimal dependencies (panda + numpy).
    '''
    def __init__(self, feature_list, feature_weights):
        '''
            feature_list: type list
                the list of features, the given testing data should follow this order
            feature_weights:  type dict
                the dictionary that associates each feature with its weight
        '''
        self.feature_list = feature_list
        self.feature_weights = feature_weights

        self.weight_vec = np.array([feature_weights[feature] for feature in feature_list])


    def get_feature_weights(self):
        return self.feature_weights
        
    def get_weight_vector(self):
        return self.weight_vec

    def get_feature_list(self):
        return self.feature_list


    def predict(self, X, binary=False):
        '''
            Give prediction for alternatives within a single session
            
            X: DataFrame, or np.ndarray
            
            return: np.ndarray
        '''
        
        dot_product = X.dot(self.weight_vec)
        if isinstance(dot_product, pd.Series):
            dot_product = dot_product.values
        
        return softmax(dot_product)


    def save(self, file_name):
        '''
            Serialize the model object into a file
        '''
        with open(file_name, mode='wb') as model_file:
            # use lower version of protocol to be compatible with Python 2
            pickle.dump(self, model_file, protocol=2)
            print('save model to ', file_name)


        