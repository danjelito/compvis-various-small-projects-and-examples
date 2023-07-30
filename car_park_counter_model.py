import os
import utils
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
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
        'solver': trial.suggest_categorical('solver', [
            'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 
            'sag', 'saga'
        ]), 

    }

    clf= LogisticRegression(**params)

    acc= cross_val_score(clf, 
                        x_train, y_train, 
                        cv= 3, scoring= 'accuracy')

    return acc.mean()

n_trials= 10
# make the sampler behave in a deterministic way
sampler = TPESampler(seed= 1) 
study = optuna.create_study(direction='maximize', sampler= sampler)
study.optimize(objective, n_trials= n_trials)
best_params= study.best_params

# train classifier with best params
classifier= LogisticRegression(**best_params)
classifier.fit(x_train, y_train)

# test
test_acc= classifier.score(x_test, y_test)
print(f'Test acc: {test_acc}')

# save model
path= Path('model/car_park_clf.pickle')
pickle.dump(classifier, open(path, 'wb'))
print('Model saved.')