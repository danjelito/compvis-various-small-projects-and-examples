import os
import utils
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
x_train, x_val, y_train, y_val = train_test_split(
    x_train, 
    y_train,
    stratify= y_train,
    test_size= 0.2,
    shuffle= True,  
    random_state= 42
)

# train classifier
classifier= SVC()
n_trials= 10

# define an objective function to be maximized
def objective(trial):

    params= {
        'C': trial.suggest_float('C', 1e-3, 1000.0, log= True),
        'gamma': trial.suggest_float('gamma', 1e-3, 100.0, log= True),
    }
    classifier.set_params(**params)
    classifier.fit(x_train, y_train)
    val_acc= classifier.score(x_val, y_val)

    return val_acc

# create a study object and optimize the objective function
sampler = TPESampler(seed= 10) # make the sampler behave in a deterministic way
study = optuna.create_study(direction='maximize', sampler= sampler)
study.optimize(objective, n_trials= n_trials)
best_params= study.best_params

# train classifier with best params
classifier= SVC(**best_params)
classifier.fit(x_train, y_train)

# test
test_acc= classifier.score(x_test, y_test)
print(f'Test acc: {test_acc}')

# save model
path= Path('output/car_park_counter_model.pkl')
pickle.dump(classifier, open(path, 'wb'))
print('Model saved.')