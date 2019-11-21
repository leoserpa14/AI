import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importing dataset directly from keras
data = keras.datasets.fashion_mnist

# splitting our data into train and test data
(train_images, train_labels), (test_images, test_labels) = data.load_data() # load_data makes it easy but to separate its because of keras library

# if we look at https://www.tensorflow.org/tutorials/keras/classification we will see that each number label means something
# print(train_labels[6])
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(train_images[45], cmap=plt.cm.binary)
# plt.show()

# print(train_images[45]) # this is actually our image (list of values of pixels grayscale)
# We want to minimize the magnitude of our data to be inside 0 and 1
train_images = train_images/255
test_images = test_images/255

# print(train_images[45])

# We need to 'Flatten' our data (transform the array of arrays into just one array)
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation="relu"), # hidden layer, with 'rectified linear unit' activation function
	keras.layers.Dense(10, activation="softmax") # our output layer, 'softmax' activation makes our neurons values add up to 1
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # Look these functions up (optimizer, loss functions, metrics)

# Time to train our model
model.fit(train_images, train_labels, epochs=7) # 'epochs' is how many times our model will see the same train data

# # Test our model
test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Acc:", test_acc)


# Next video starts now: We will learn how to use the model we just trained

# prediction = model.predict(np.array())
prediction = model.predict(test_images)
# If I wanted to predict for only one item I should put it in a list
# prediction = model.predict([test_images[7]])


# # Use numpy function to find the highest value of the array and give us the index
# print(class_names[int(np.argmax(prediction[0]))]) # the model thinks it's an Ankle Boot
# or

for i in range(15):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction: " + class_names[int(np.argmax(prediction[i]))])
	plt.show()




