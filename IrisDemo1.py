import pandas
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    url = "irisdata.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    print("\n")
    print(dataset.describe())  # descriptive statistics of the dataset like mean, std, min, max, etc.
    print("\n")
    print(dataset.groupby('class').size())  # count of each class
    print("\n")
    # dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # plot each class, with a box plot and 2 subplots and 2 rows and 2 columns not sharing the coordinates
    # plt.show()
    # dataset.hist()
    # generate histogram for each attribute
    # plt.show()
    # scatter_matrix(dataset)
    # scatter matrix for each attribute
    # plt.show()

    array = dataset.values
    x = array[:, 0:4]
    y = array[:, 4]
    validation_size = 0.2
    seed = 6
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                        random_state=seed)

    '''
    This code snippet is using the scikit-learn library to split a dataset into training and testing sets for use in machine learning.

The dataset is first converted to a NumPy array using the `values` attribute of a pandas DataFrame or Series.
 Then, the features of the dataset are extracted into the variable `x` using NumPy indexing.
  The target variable is extracted into `y`.

The `train_test_split` function from the `model_selection` module is used to randomly split the dataset into training and testing sets.
 The `test_size` parameter is set to 0.2, indicating that 20% of the data should be used for testing, while the remaining 80% is used for training.
  The `random_state` parameter is set to 6, which ensures that the random splitting of the data is reproducible.

The resulting training and testing sets are stored in the variables `x_train`, `x_test`, `y_train`, and `y_test`, which can then be used as input for a machine learning model.
    '''

    scoring = 'accuracy'

    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=1000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evalua te each model in tun
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) " % (name, cv_results.mean(), cv_results.std())
        print(msg)

# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()
'''


The code snippet seems to be implementing a machine learning pipeline for classification using various algorithms. 

- The first six lines define the models to be used: 
Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Naive Bayes, and Support Vector Machines.
- The models are then evaluated using cross-validation with 10 folds using the `model_selection.KFold` and `model_selection.cross_val_score` functions from scikit-learn.
- The `scoring` variable is not defined in the code, so it's unclear what metric is being used to evaluate the models.
- The `results` and `names` lists are used to store the cross-validation results and the names of the models, respectively.
- Finally, the code prints the mean and standard deviation of the cross-validation scores for each model. 

Overall, this code provides a good starting point for evaluating the performance of multiple classification models on a given dataset.
'''
...
# Make predictions on validation dataset
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print("accuracyScore" ,accuracy_score(Y_test, predictions))
print("\n")
print("confusionMatrix \n", confusion_matrix(Y_test, predictions))
print("\n")
print("classificationReport\n" ,classification_report(Y_test, predictions))
print("\n")
