"""
   Module Name:  main.py
   Company Name: LAN Innovations
   Author Names: Sabarish K and Kavin D
   Last modified: 30-November-2021
   Description:
                ML model Creation for IAssist
                -----------------------------
   This python module is a main file used to create a Recurrent Neural Network(RNN) sequential model or simply it may
   called machine learning model for the given human conversation data to build a human machine chat or a Chatbot
   application. This machine learning model can be used in a backend side to provide a solution to the users questions
   on front side. This is a classification type model to categorize each different kinds of conversation data into a
   single trained model. Only use .CSV format to create a model.
   Steps involved below:
   1. Load cleansed chat datasets into pandas dataframe
   2. Separate error messages with and without duplicates, Resolutions
   3. Clean error messages
   4. Word Tokenizer
   5. Change error message sequence index into maximum length
   6. OneHot Encoding
   7. Create sequential model
   8. Plot the model results
   9. Model prediction
   For more details check Github link below:
   https://github.com/kavindevarajan/IT_support_chatbot
"""
import re
import numpy as np
import pandas as pd
import asyncio
import time
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import matplotlib.pyplot as plt
# from IPython import get_ipython
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('C:\\inetpub\\wwwroot\\file_upload\\model-creation-status.log')
file_handler.setFormatter(formatter)

# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# app.logger.addHandler(stream_handler)


async def load_dataset(file_input):
    """Define a method to load the dataset in.csv format and separate Error messages,
    Resolutions and Unique error messages(without duplicates)."""
    logger.debug("CSV file ready to pick")
    time.sleep(5)
    logger.debug('delay success')
    df = pd.read_csv(file_input, encoding="latin1", names=["Error_Message", "Resolution"])
    logger.debug("file loaded")
    resolution1 = df["Resolution"]
    unique_resolution1 = list(set(resolution1))
    unique_counts = len(unique_resolution1)
    error_message1 = list(df["Error_Message"])
    return resolution1, unique_resolution1, error_message1, unique_counts


logger.debug("CSV file ready to load")
# Load the dataFrame
resolution, unique_resolution, error_message, unique_counts1 = asyncio.run(load_dataset("C:\\inetpub\\wwwroot\\"
                                                                                        "iAssist_IT_support\\Datasets"
                                                                                        "\\chatDataset_all.csv"))
logger.debug("CSV file loaded successfully")
# Convert resolution as string type
resolution = resolution.astype(str)
# Convert unique values as string and store it into a list
unique_resolution = list(map(str, unique_resolution))


# print(unique_resolution)
# print(unique_counts1)


def cleaning(sentences):
    """Define a method for cleaning and tokenizing the data in Error_Message using NLTK"""
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-zA-Z0-9]', " ", s)
        w = word_tokenize(clean)
        words.append([i.lower() for i in w])
    return words


cleaned_words = cleaning(error_message)


def create_tokenizer(words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    """Define a Keras preprocessing called 'tokenizer' method from tensorflow to updates internal vocabulary based
    on a list of texts, lower case all the data and tokenize from words to sequence data."""
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def max_length(words):
    """Define function to find maximum length of message presence in error_message series"""
    return len(max(words, key=len))


word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
# Maximize all error message sequence data with maximum index value using 0's
max_length = max_length(cleaned_words)


# print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


def encoding_doc(token, words):
    """Define function to convert all the tokenized texts into sequences list of list."""
    return token.texts_to_sequences(words)


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


def padding_doc(encoded, maxlength):
    """Define function to pad all the sequences to get same length"""
    return pad_sequences(encoded, maxlen=maxlength, padding="post")


# Pad the sequence using text to sequence values and maximum index length
padded_doc = padding_doc(encoded_doc, max_length)

# tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_resolution, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
wordIndex = output_tokenizer.word_index
encodedOutput = encoding_doc(output_tokenizer, resolution)

# Reshape an encoded output into a single length of array.
encodedOutput = np.array(encodedOutput).reshape(len(encodedOutput), 1)


def one_hot(encode):
    """Define an OneHotEncoding function to convert all the categorical data into sequential data.
    It will be used to train the model."""
    o = OneHotEncoder(sparse=False)
    return o.fit_transform(encode)


# encodedOutput = np.array([np.array(xi) for xi in encodedOutput])
outputOneHot = one_hot(encodedOutput)
OneHotShape = outputOneHot.shape

# Use train_test_split class from scikit learn module to separate dataset for training and testing
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, outputOneHot, shuffle=True, test_size=0.2)


# print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
# print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


def create_model(vocabsize, maxlength):
    """Build the Sequential model with Embedding layer, Long Short-Term Memory layer, Bidirectional wrapper for RNNs,
    Activations functions relu and softmax for hidden layers"""
    fmodel = Sequential()
    fmodel.add(Embedding(vocabsize, 128, input_length=maxlength, trainable=False))
    fmodel.add(Bidirectional(LSTM(128)))
    fmodel.add(Dense(32, activation="relu"))
    fmodel.add(Dropout(0.1))
    fmodel.add(Dense(unique_counts1, activation="softmax"))
    return fmodel


model = create_model(vocab_size, max_length)

"""Compile the model with some parameters
    1. Categorical crossentropy is a loss function that is used in multi-class classification tasks.
    2. Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. 
    3. Metrics accuracy to calculates how often predictions equal labels."""
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Print the created and compiled model summary or conclusion
model.summary()

# Create a .h5 file to load the output of model
filename = 'C:\\inetpub\\wwwroot\\iAssist_IT_support\\model\\model_gitex1.h5'

# earlyStop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
# Set the checkpoint to stop training the model if the value loss increase
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

logger.debug("New model is ready to train")
# Now finally train the model with train data and validation data and store the results in a variable.
history = model.fit(train_X, train_Y, epochs=100, batch_size=32,
                    validation_data=(val_X, val_Y), callbacks=[checkpoint])

logger.debug("New model training is done")
# load the trained model in Hierarchical Data Format 5 (.h5).
modelFile = load_model("C:\\inetpub\\wwwroot\\iAssist_IT_support\\model\\model_gitex1.h5")

# An average values of train and validation data from overall epochs
train_loss, train_acc = modelFile.evaluate(train_X, train_Y, verbose=0)
test_loss, test_acc = modelFile.evaluate(val_X, val_Y, verbose=0)
logger.debug(f'Accuracy and loss of the best model : ')
logger.debug(f'Train accuracy: {train_acc * 100:.3f} % || Test accuracy: {test_acc * 100:.3f} %')
logger.debug(f'Train loss: {train_loss:.3f} || Test loss: {test_loss:.3f}')

# Convert the model into a TensorflowLite model for offline mobile applications.
logger.debug("TensorflowLite model creation process started")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tfLiteModel = converter.convert()

# Save the model.
logger.debug("TensorflowLite file preparing")
with open('C:\\inetpub\\wwwroot\\iAssist_IT_support\\model\\model_gitex_TFLite1.tflite', 'wb') as file:
    file.write(tfLiteModel)
logger.debug("TensorflowLite model created successfully")


# # Visualize the model results
# # Train data graph plots
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.title('Model loss')
plt.ylim([0, 2])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(['train', 'test'], loc='upper right')
plt.savefig("C:\\inetpub\\wwwroot\\iAssist_IT_support\\Model_results\\Model-loss-2.4.png")

# Validation data graph plots
plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.title('Model accuracy')
plt.ylim([0, 1])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(['train', 'test'], loc='lower right')
plt.savefig("C:\\inetpub\\wwwroot\\iAssist_IT_support\\Model_results\\Model-accuracy-2.4.png")


def predictions(textin):
    """Predict the confidential level of all unique resolutions for
    an error message sent by user and then return all prediction levels"""
    clean = re.sub(r'[^ a-zA-Z0-9]', " ", textin)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]

    # Tokenize the text data into a sequence
    test_ls = word_tokenizer.texts_to_sequences(test_word)

    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    test_ls = np.array(test_ls).reshape(1, len(test_ls))

    # Convert given error message's numerical index value into a maximum index value by adding zero's
    x = padding_doc(test_ls, max_length)
    prediction_out = modelFile.predict(x)
    return prediction_out


def get_final_output(predin, classes):
    """Define function to print the confidence level of individual classes"""
    # Pick a numerical data from list of list
    predict = predin[0]

    # Convert a unique resolutions from list to numpy array
    classes = np.array(classes)

    # Sort the prediction values based on list index positions
    ids = np.argsort(-predict)

    # Change all the values in unique resolution's index positions as same as prediction values
    classes = classes[ids]

    # Now sort the prediction values in descending orders
    predict = -np.sort(-predict)
    for i in range(predin.shape[1]):
        print("%s has confidence = %s" % (classes[i], (predict[i])))


text = "BAPI call must have a SAVE or DIALOG method"
prediction = predictions(text)
get_final_output(prediction, unique_resolution)
logger.debug("Script ended successfully")

# Erase all the variables at end of execution
# get_ipython().magic('reset -f')
