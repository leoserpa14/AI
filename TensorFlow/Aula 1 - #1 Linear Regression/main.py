import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle # para salvar nossos modelos desejados
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# The attribute that we want to predict is the G3 (Label: the attribute we're trying to get)
predict = "G3"

# This is gonna return to us a new data frame without G3
x = np.array(data.drop([predict], 1)) # All of our attributes
y = np.array(data[predict]) # All of our labels

# Split our attributes and labels into 4 different arrays, so the machine won't be able access our original database and
# determine the outcome based on its patterns
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# So we're gonna have the machine to train on a smaller portion of the data, the train data - in this case a 10%
# portion, and compare with the 'test' data, which will not have the 'train' data

best = 0
# for _ in range(100):
#
# 	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
# 	# Create a linear training model
# 	linear = linear_model.LinearRegression()
#
# 	linear.fit(x_train, y_train) # Fit a linear regression with our train data
# 	acc = linear.score(x_test, y_test) # Returns the accuracy of our model
# 	print(acc)
#
# 	if acc > best:
# 		best = acc
# 		# Salvar esse nosso modelo
# 		with open("studentmodel.pickle", "wb") as f:
# 			pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Angular coefficients: ", linear.coef_) # Coeficientes angulares da linha para um ambiente de 5 dimensões (5 atttributes)
print("Linear coefficient: ", linear.intercept_) # Coeficiente linear

predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print(predictions[x], x_test[x], y_test[x]) # (grade our model predicted, array in which it was based, actual grade)

# Fazer o gráfico para uma variável e minha label desejada
p = "absences"
style.use("ggplot")
pyplot.scatter(x=data[p], y=data["G3"]) # dou o x e o y da função
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()