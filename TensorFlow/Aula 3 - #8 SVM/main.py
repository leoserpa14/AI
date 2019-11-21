import sklearn
from sklearn import datasets
from sklearn import svm # SVM is used to classify data
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer() # breast cancer dataset

# print(cancer.feature_names) # attributes?
# print(cancer.target_names)

x = cancer.data
y = cancer.target # 0 represents malignant, 1 represents benign

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train, y_train)
classes = ['malignant', 'benign']

# Support Vector Classification
clf = svm.SVC(kernel="linear") # list of kernels: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Changed kernel from rbf to linear, much better results
# 'poly' - Polinomial - demora para um caralho (exponencial) // Posso usar 'poly' com 'degree=2'

# clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred) # Compare these two lists
print(acc)