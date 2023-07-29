from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import optuna
import pickle
import numpy as np

path= 'dataset/american sign language digit/sign_lang_dataset.pickle'
dataset= pickle.load(file= open(path, 'rb'))

x= dataset['features']
y= dataset['labels']

# get only images where hand is detected
mask= np.sum(x, axis= 1) > 0
x= x[mask]
y= y[mask]


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2, random_state=42)


def objective(trial):
    
    params= {
        'n_estimators': trial.suggest_int('n_estimators', 2, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 400),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
    }

    clf= RandomForestClassifier(**params)

    f1= cross_val_score(clf, x, y, cv= 3, scoring= 'f1_weighted')

    return f1.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params= study.best_params

clf= RandomForestClassifier(**best_params)
clf.fit(x_train, y_train)

y_pred= clf.predict(x_test)
acc= accuracy_score(y_test, y_pred)
f1= f1_score(y_test, y_pred, average= 'weighted')

print(f'acc = {acc: .4f} - f1 = {f1: .4f}')

f= open('model/sign_digit.pickle', 'wb')
pickle.dump({'model': clf}, f)
f.close()