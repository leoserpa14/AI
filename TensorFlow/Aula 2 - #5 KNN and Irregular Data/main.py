import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("car.data")
# print(data.head())

# preprocessing: Help us convert n, on-numerical data to numerical data (integers)
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"])) # This will put our 'buying' values in a list and transform them to ints
maint = le.fit_transform(list(data["maint"])) # Also, this returns a numpy array
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(buying)

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety)) # This converts our 5 attributes into just one big list
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print("\n", x_train, "\n", y_test)

model = KNeighborsClassifier(n_neighbors=7) # Our model receives 1 parameter = number of neighbours

model.fit(x_train, y_train) # Train our model with x_train and y_train
acc = model.score(x_test, y_test) # Compare the score with x_test and y_test
print(acc, "\n")
# If my hyperparameter (k) is too high, we start losing accuracy, as explained in the video
# https://www.youtube.com/watch?v=vwLT6bZrHEE

predicted = model.predict(x_test) # Predicted results of our model based on x_test, received in a list
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(predicted)):
	print("Predicted: ", names[predicted[i]], " Based on Data: ", x_test[i], "Actual result: ", names[y_test[i]])
	# we can take the 'names' out of this function and we will receive the results in numbers
	n = model.kneighbors([x_test[i]], 7, True)
	print("N: ", n)