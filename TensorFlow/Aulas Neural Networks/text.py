import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) # vamos só pegar as 10000 palavras que mais aparecem, e não todas as palavras

# print(test_data[0]) # we can see that each word actually is an int

word_index = data.get_word_index() # jeito que o tutorial do tensorflow fala pra transformar nossa database em palavras
word_index = {k: (v+3) for k, v in word_index.items()} # adicionando 3 para todas as int
word_index["<PAD>"] = 0 # vamos usar padding para transformar todas as reviews em itens de mesmo tamanho
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()]) # swap all the values and the keys, because we want the integer pointing to a word

# Fazer isso provavlmente seria muito mais difícil se o Keras não tivesse essa função pronta para esse caso
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=512)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=512)

# print("\n", len(train_data[0]), len(test_data[54]))

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(test_data[0]))
#
# print("\n", len(test_data[0]), len(test_data[6]))  # mostrar que temos tamanhos diferentes de reviews
# # Precisamos que nossos inputs tenham o mesmo tamanho. Resolvemos isso na linha 21 com uma função já feita pelo tensorflow.keras
# # parei o vídeo em 17:06


# model down here
""" Agora que já treinei meu modelo, posso comentar isso aqui e só carregar o model.h5
model = keras.Sequential()
# Contexto e palavras similares importam, então como vamos resolver isso? Assim:
model.add(keras.layers.Embedding(88000, 16)) # Embedding > Word vectors // Criamos 10.000 word vectors randomly, each represents a word
model.add(keras.layers.GlobalAveragePooling1D()) # Transforms the dimension our data is in into a lower dimension (more computational efficient)
model.add(keras.layers.Dense(16, activation="relu"))
# We want the model to tell us if the review is good or bad, so 1 neuron that can receive a 0 or a 1
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# loss function binary_crossentropy will calculate the difference between the result and the expected result
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split our training data into validation data and actual training data
x_val = train_data[:10000] # 10.000 out of 20.000?
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1) # search what verbose is

results = model.evaluate(test_data, test_labels)

# test_review = test_data[0]
# predict = model.predict([test_review])
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction: " + str(predict[0]))
# print("Actual: " + str(test_labels[0]))
print(results)

model.save("model.h5") # h5 is the extension. # TODO: look it up what it means
"""


def review_encode(s):  # 's' being a list of words
	encoded = [1]  # pra começar com <START>

	for word in s:
		if word.lower() in word_index:  # se a palavra já existir no dicionário com 88000 palavras
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)  # senão, adicionar o <UNK>

	return encoded


model = keras.models.load_model("model.h5")


with open("lionkingreview.txt", encoding="utf-8") as f:
	for line in f.readlines():
		# preprocess our file
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=512)
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])
