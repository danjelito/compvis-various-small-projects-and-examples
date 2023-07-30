import os
import utils
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.samplers import TPESampler
import numpy as np
import pickle

# get images and labels
image_dir= Path('dataset/parking lot/image')
images, labels= utils.load_image_dataset(image_dir, shuffle= False)

# convert images from path to images
images= [imread(i) for i in images]

# resize image to 15x15
images= [resize(i, (15, 15)) for i in images]

# flatten image to 1-d array
# convert labels to array
images= np.asarray([i.flatten() for i in images])
labels= np.asarray(labels)

# train val test split
x_train, x_test, y_train, y_test = train_test_split(
    images, 
    labels,
    stratify= labels,
    test_size= 0.2,
    shuffle= True,  
    random_state= 42
)

def objective(trial):
    
    params= {
        'C': trial.suggest_float('C', 0.00001, 10, log= True),
        'kernel': trial.suggest_categorical('kernel', [
            'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        ]), 
    }
    clf= SVC(**params)
    acc= cross_val_score(clf, 
                         x_train, y_train, 
                         cv= 3, scoring= 'accuracy')
    return acc.mean()

# HP tuning
n_trials= 1
sampler = TPESampler(seed= 1) # set random state from sampler
study = optuna.create_study(direction='maximize', sampler= sampler)
study.optimize(objective, n_trials= n_trials)
best_params= study.best_params

# train classifier with best params
clf= SVC(**best_params)
clf.fit(x_train, y_train)

# test accuracy
y_pred= clf.predict(x_test)
test_acc= accuracy_score(y_test, y_pred)

# test F1
label_map= {'empty': 1, 'not_empty': 0}
y_test_mapped= np.vectorize(label_map.get)(y_test)
y_pred_mapped= np.vectorize(label_map.get)(y_pred)
test_f1= f1_score(y_test_mapped, y_pred_mapped)

# print test result
print(f'Test acc: {test_acc: .4f} - Test F1: {test_f1: .4f}')

# save model
path= Path('model/car_park_clf.pickle')
pickle.dump(clf, open(path, 'wb'))
print('Model saved.')