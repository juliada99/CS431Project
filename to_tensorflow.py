import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Location of the csv data
filename = './data.csv'

# Load raw data as a pandas Dataframe (note that this is not immediately usable for a tensorflow
# model)
data = pd.read_csv(filename, index_col = 0)
data.head()

data_features = data.copy()
# Use confirmed as labels
data_labels = data_features.pop('Confirmed')

# ---------------------------------------------------------
# We build a model that implements the preprocessing logic using Keras functional API
# ---------------------------------------------------------

# Build a set of symbolic keras.Input objects, matching the names and data-types of the CSV columns.
inputs = {}

for name, column in data_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

# Concatenate the numeric inputs together, and run them through a normalization layer:
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(data[list(numeric_inputs.keys())]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

# Collect all the symbolic preprocessing results, to concatenate them later.
preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():

  # We are only interested in string inputs
  if input.dtype == tf.float32:
    continue

  # Map from strings to integer indices in a vocabulary
  lookup = layers.StringLookup(vocabulary=np.unique(data_features[name]))

  # Convert the indexes into float32 data appropriate for the model.
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  # We created a one-hot vector for each input.
  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

# Concatenate all the preprocessed inputs together, and build a model that handles the preprocessing:
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = data_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

# Convert the Pandas Dataframe to a dictionary of tensors:
data_features_dict = {name: np.array(value) 
                         for name, value in data_features.items()}

# Return a tf.data.Dataset that implements a generalized version of the slices function, 
# in TensorFlow. 
features_ds = tf.data.Dataset.from_tensor_slices(data_features_dict)

for example in features_ds:
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break

# Make a dataset of (features_dict, labels) pairs:
data_ds = tf.data.Dataset.from_tensor_slices((data_features_dict, data_labels))

# Shuffle and batch the data.
data_batches = data_ds.shuffle(len(data_labels)).batch(32)

# Build the model on top of this:
def data_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

data_model = data_model(data_preprocessing, inputs)

# Pass the dataset:
data_model.fit(data_batches, epochs=5)


