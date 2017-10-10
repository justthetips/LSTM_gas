# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from typing import Tuple



def describe(ds: pandas.DataFrame):
    print(ds.describe())
    print(ds.groupby('class').size())


def basic_charts(ds: pandas.DataFrame):
    # box and whisker plots
    ds.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    # histograms
    ds.hist()
    plt.show()

    # scatter plot matrix
    scatter_matrix(ds)
    plt.show()


def get_validation_dataset(ds: pandas.DataFrame, validation_size: float = 0.20, seed: int = 7):
    # Split-out validation dataset
    array = ds.values
    X = array[:, 0:4]
    Y = array[:, 4]
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


def evaulate_models(X_train:np.array, Y_train:np.array, seed:int=7) -> np.array:
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return results, names

def plot_algo_results(results:list, names:list):
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def train_knn(X_train, X_validation, Y_train, Y_validation):
    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))





# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
scoring = 'accuracy'

print(dataset.shape)

X_train, X_validation, Y_train, Y_validation = get_validation_dataset(dataset)
plot_algo_results(*evaulate_models(X_train, Y_train))
train_knn(X_train, X_validation, Y_train, Y_validation)


