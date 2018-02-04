'''
This code is written by Vishnu deo gupta, The main purpose of the code is to calculate the importance of a
feature in feature sets. As the datasets is pretty small i ran the fitting and testing of data 25 times to calculate the
mean of accuracy.
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
#from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
#from scipy import stats
#import pickle

df = pd.read_csv('pima-indians-diabetes.data.txt')
print(df.head())




'''
Firstly i was not including the blood_presuure feature. the overall accuracy was 74%.
but now the accuracy is more then 75%, its approx 77%

Although serum_insulin is great contributer... contributes more than skin_fold_thickness..

'''



try_attributes = ['Diastolic_blood_press']


# these may be the attributes:--- 'skin_fold_thickness', 'serum_insulin',

drop_attributes = [ 'skin_fold_thickness', 'serum_insulin','Class']

X = np.array(df.drop(drop_attributes, 1).astype(float))

X = preprocessing.scale(X)
y = np.array(df['Class'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#model = neighbors.KNeighborsClassifier()
acc = 0
for i in range(25):
    model = neighbors.KNeighborsClassifier()
    model.fit(X_train, y_train)
    '''
    The following line ( model.feature_importances_ ) is most important in this code, it helps deciding the feature sets. 
    It computes the importance of any feature. It is quite helpfull, specially on this problem, as initially
    i was adding the feature serum_insulin, ( as it sounds biological ) but that reduced the accuracy to 69%. 
    And computation of importance of features showed that it was the least impotant feature. It helps to reduce the overfit.
    '''
    #print(model.feature_importances_)

    accuracy = model.score(X_test, y_test)
    acc += accuracy
print(acc/25)

