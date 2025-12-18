######################################################
#  Student Name: Devin Bailie
#  Student ID: bailied2 / W30644462
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: 12/9/2025
#  GitHub Link: https://github.com/bailied2/csci460-mri-tumor-detection
######################################################


import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses TensorFlow warnings


import tensorflow as tf
import numpy as np
import time

import sklearn.model_selection as sklearnmodels


def readImageDirectory(baseDir, classDict, trainRatio=0.6, testRatio=0.2, imageSize=None):
  """
  Take a base directory and a dictionary of class values,
  go into that directory and read all the images there to
  build an image data set.  It is assumed that different
  classes are in different subdirectories with that class
  name. Split data according to trainSize and testSize, 
  then return the corresponding image tensors, as well as  
  the label Y vectors.  If a tuple (x,y) imageSize is given,
  then enforce all the images are of a specific size.
  """
  # Check that trainRatio and testRatio are valid
  if trainRatio + testRatio >= 1.0:
    raise ValueError("trainRatio + testRatio must be < 1.0")
  
  # Initialize the X tensor and the Y vector raw structures
  imagesX = []
  imagesY = []

  for classKey in classDict:
    dirName = os.path.join(baseDir, classKey)
    for filename in os.listdir(dirName):
        # Filename name is the base name + class name + image file name
        fn = os.path.join(dirName, filename)

        # Load the image, then make sure it is scaled all all three color channels
        rawImage = tf.keras.preprocessing.image.load_img(fn, target_size=imageSize)
        image = tf.keras.preprocessing.image.img_to_array(rawImage)/255.0

        # Grow the image tensor and the class vector by 1 entry
        imagesX.append(image)
        imagesY.append(classDict[classKey])

  imagesY = np.array(imagesY, dtype="float32")

  # First split data into TR (training) and temp, remainder will be split again
  X_TR, X_temp, Y_TR, Y_temp = sklearnmodels.train_test_split(
     imagesX, 
     imagesY, 
     test_size=1.0 - trainRatio, 
     random_state=42, 
     stratify=imagesY
  )
  # Now split remaining data into TT (testing) and VL (validation)
  X_TT, X_VL, Y_TT, Y_VL = sklearnmodels.train_test_split(
     X_temp, 
     Y_temp, 
     test_size = testRatio/(1.0 - trainRatio), 
     random_state=42, 
     stratify=Y_temp
  )
  # Return data sets with X converted to tensors and Y converted to 'hot-ones'
  return (
     tf.convert_to_tensor(X_TR), 
     tf.convert_to_tensor(X_TT),
     tf.convert_to_tensor(X_VL),
     tf.keras.utils.to_categorical(Y_TR),
     tf.keras.utils.to_categorical(Y_TT),
     tf.keras.utils.to_categorical(Y_VL),
  )

def buildCandidateCNN(config):
  """
  Builds and return a Sequential CNN model using the 
  given config dictionary.
  """
  # Start the model
  model = tf.keras.models.Sequential()

  # Add input layer to constrain image tensors to correct size
  model.add( tf.keras.Input(shape=(config['imageSize'], config['imageSize'], 3)) )

  # Build convolutional layers using supplied parameters:
  for _ in range(config['numConv2DLayers']):
    model.add( tf.keras.layers.Conv2D(config['numFilters'], # use numFilters param for the layer
                                      (config['kernelSize'], config['kernelSize']), # use kernelSize param
                                      activation="relu",) )
    # Add 2D pooling layer for each Conv2D layer:
    model.add( tf.keras.layers.MaxPooling2D((config['poolingSize'], config['poolingSize'])) ) # use poolingSize param
  

  # Flatten image tensors to 1D vectors to prepare for MLP
  model.add( tf.keras.layers.Flatten() )

  # Make a middle layer using denseNodes parameter:
  model.add( tf.keras.layers.Dense(config['denseNodes'], activation="relu") )

  # Make the output size 2. Let's softmax the activation so
  # we get probabilities in the end.
  model.add( tf.keras.layers.Dense(2, activation="softmax") )

  return model

# Function for computing model fitness score from accuracy and time results
def computeFitness(accuracy, trainingTime, alpha=0.7):
  normalized_time = trainingTime / 1000.0 # Normalize training time by dividing by 1000
  # Compute fitness as weighted combination of accuracy and inverse of training time
  return alpha * accuracy + (1 - alpha) * (1 / (1 + normalized_time))


# Set seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

# Dictionary holding lists of each value that will be tested for each hyperparameter on the validation set.
possibleValues = {
  'numFilters': [32, 50, 64, 80],
  'kernelSize': [2, 3, 4, 5],
  'numConv2DLayers': [1, 2, 3],
  'poolingSize': [2, 3, 4],
  'denseNodes': [16, 20, 24, 32, 50],
  'learningRate': [0.005, 0.01, 0.02, 0.05]
}

# Dictionary for storing default hyperparameters, to be updated as validation tests are performed.
cnnConfig = {
  'numFilters': 50,
  'kernelSize': 4,
  'numConv2DLayers': 1,
  'poolingSize': 2,
  'denseNodes': 20,
  'learningRate': 0.01,
}
# Use Categorical Cross-entropy loss function (for hot-ones style predictions)
lossFunction = tf.keras.losses.CategoricalCrossentropy()

# Dictionary for each dataset (one for each image size)
datasets = {}

# Dictionary for storing test results for each imageSize
results = {
  'imageSize': {},
  'numFilters': {},
  'kernelSize': {},
  'numConv2DLayers': {},
  'poolingSize': {},
  'denseNodes': {},
  'learningRate': {}
}

# First, we'll test image size separately since it involves preprocessing images

imageSizes = [128, 224, 256] # Sizes to test

for size in imageSizes:
  # Call readImageDirectory to preprocess image data for CNN with given size
  datasets[size] = readImageDirectory(
    "/data/csci460/BTD",
    {"yes":1, "no":0},
    trainRatio=0.6,
    testRatio=0.2,
    imageSize=(size, size),
  )

  X_TR, X_TT, X_VL, Y_TR, Y_TT, Y_VL = datasets[size]

  # Update cnnConfig with image size value
  cnnConfig['imageSize'] = size

  model = buildCandidateCNN(cnnConfig)

  # Set optimizer as a stochastic gradient descent method with
  # a learning rate of 0.01
  opt = tf.keras.optimizers.SGD(learning_rate=0.01)

  # Set the model for training
  model.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

  start = time.time() # Get current time

  # Perform the induction
  model.fit(X_TR, Y_TR, epochs=10, verbose=0)

  # Initialize results dict for current size
  results['imageSize'][size] = {}
  
  # Store time elapsed during training
  results['imageSize'][size]['time'] = time.time() - start

  # Evaluate on validation set and store results
  results['imageSize'][size].update(model.evaluate(X_VL, Y_VL, verbose=0, return_dict=True))

  # Get per-class accuracy values
  Y_pred_probs = model.predict(X_VL, verbose=0) # Get predicted probabilities
  Y_pred = np.argmax(Y_pred_probs, axis=1) # Map each prediction to higher probability
  Y_true = np.argmax(Y_VL, axis=1) # Perform same function on Y_VL so its format matches

  # Calculate per-class accuracies and store results
  results['imageSize'][size]['noAccuracy'] = np.mean(Y_pred[Y_true == 0] == Y_true[Y_true == 0])
  results['imageSize'][size]['yesAccuracy'] = np.mean(Y_pred[Y_true == 1] == Y_true[Y_true == 1])

  # Compute overall and per-class fitness
  results['imageSize'][size]['fitness'] = computeFitness(
    results['imageSize'][size]['accuracy'],
    results['imageSize'][size]['time']
  )
  results['imageSize'][size]['noFitness'] = computeFitness(
    results['imageSize'][size]['noAccuracy'],
    results['imageSize'][size]['time']
  )
  results['imageSize'][size]['yesFitness'] = computeFitness(
    results['imageSize'][size]['yesAccuracy'],
    results['imageSize'][size]['time']
  )

  # Log test performed
  print(f"Image Size: {size}x{size} - \n\
        Accuracy: {results['imageSize'][size]['accuracy']} \n\
        'Yes' Accuracy: {results['imageSize'][size]['yesAccuracy']} \n\
        'No' Accuracy: {results['imageSize'][size]['noAccuracy']} \n\
        'Yes' Fitness: {results['imageSize'][size]['yesFitness']} \n\
        'No' Fitness: {results['imageSize'][size]['noFitness']} \n\
        Training Time: {results['imageSize'][size]['time']} \n\
        Fitness: {results['imageSize'][size]['fitness']}")

# Now we can pick the best-performing image size from the initial test
bestFitness = None
for size, sizeResults in results['imageSize'].items():
  if bestFitness is None or sizeResults['fitness'] > bestFitness:
    cnnConfig['imageSize'] = size
    bestFitness = sizeResults['fitness']

print(f"Image size {cnnConfig['imageSize']}x{cnnConfig['imageSize']} chosen")

# Now that we have chosen the imageSize, we can stick to a single dataset.
X_TR, X_TT, X_VL, Y_TR, Y_TT, Y_VL = datasets[cnnConfig['imageSize']]

# Now loop through each hyperparameter in possibleValues
for key, values in possibleValues.items():
  # Loop through each possible value for the current hyperparameter
  for value in values:
    # Update corresponding config value
    cnnConfig[key] = value

    # Build candidate CNN using the updated cnnConfig
    model = buildCandidateCNN(cnnConfig)

    # Set the SGD optimizer's learning rate using value from cnnConfig
    opt = tf.keras.optimizers.SGD(learning_rate=cnnConfig['learningRate'])

    # Set the model for training
    model.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

    start = time.time() # Get current time

    # Perform the induction
    model.fit(X_TR, Y_TR, epochs=10, verbose=0)

    # Initialize results dict for current value
    results[key][value] = {}
    
    # Store time elapsed during training
    results[key][value]['time'] = time.time() - start

    # Evaluate on validation set and store results
    results[key][value].update(model.evaluate(X_VL, Y_VL, verbose=0, return_dict=True))

    # Get per-class accuracy values
    Y_pred_probs = model.predict(X_VL, verbose=0) # Get predicted probabilities
    Y_pred = np.argmax(Y_pred_probs, axis=1) # Map each prediction to higher probability
    Y_true = np.argmax(Y_VL, axis=1) # Perform same function on Y_VL so its format matches

    # Calculate per-class accuracies and store results
    results[key][value]['noAccuracy'] = np.mean(Y_pred[Y_true == 0] == Y_true[Y_true == 0])
    results[key][value]['yesAccuracy'] = np.mean(Y_pred[Y_true == 1] == Y_true[Y_true == 1])

    # Compute overall and per-class fitness
    results[key][value]['fitness'] = computeFitness(
      results[key][value]['accuracy'],
      results[key][value]['time']
    )
    results[key][value]['noFitness'] = computeFitness(
      results[key][value]['noAccuracy'],
      results[key][value]['time']
    )
    results[key][value]['yesFitness'] = computeFitness(
      results[key][value]['yesAccuracy'],
      results[key][value]['time']
    )

    # Log test performed
    print(f"Parameter Tested: {key}, Value: {value} - \n\
          Accuracy: {results[key][value]['accuracy']} \n\
          'Yes' Accuracy: {results[key][value]['yesAccuracy']} \n\
          'No' Accuracy: {results[key][value]['noAccuracy']} \n\
          'Yes' Fitness: {results[key][value]['yesFitness']} \n\
          'No' Fitness: {results[key][value]['noFitness']} \n\
          Training Time: {results[key][value]['time']} \n\
          Fitness: {results[key][value]['fitness']}")

  # Now that each value has been tested, we'll choose a final value from results:
  finalValue = None
  bestFitness = None
  for v, vResults in results[key].items():
    if finalValue is None or vResults['fitness'] > bestFitness:
      finalValue = v
      bestFitness = vResults['fitness']

  # Update cnnConfig with final value
  cnnConfig[key] = finalValue
  print(f"Value '{finalValue}' chosen for hyperparameter {key}")

# Now that validation tuning is complete, we can build the final model.
model = buildCandidateCNN(cnnConfig)

# Set the optimizer and learning rate
opt = tf.keras.optimizers.SGD(learning_rate=cnnConfig['learningRate'])

# Set the model for training
model.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

# Combine TR and VL sets to train final model
X_TRVL = tf.concat([X_TR, X_VL], axis=0)
Y_TRVL = tf.concat([Y_TR, Y_VL], axis=0)

start = time.time() # Get current time

# Perform the induction, this time using both training and validation sets
model.fit(X_TRVL, Y_TRVL, epochs=10, verbose=0)

# Initialize final results dict
results['final'] = {}

# Store time elapsed during training
results['final']['time'] = time.time() - start

# Evaluate on validation set and store results
results['final'].update(model.evaluate(X_TT, Y_TT, verbose=0, return_dict=True))

# Get per-class accuracy values
Y_pred_probs = model.predict(X_TT, verbose=0) # Get predicted probabilities
Y_pred = np.argmax(Y_pred_probs, axis=1) # Map each prediction to higher probability
Y_true = np.argmax(Y_TT, axis=1) # Perform same function on Y_TT so its format matches

# Calculate per-class accuracies and store results
results['final']['noAccuracy'] = np.mean(Y_pred[Y_true == 0] == Y_true[Y_true == 0])
results['final']['yesAccuracy'] = np.mean(Y_pred[Y_true == 1] == Y_true[Y_true == 1])

# Compute overall and per-class fitness
results['final']['fitness'] = computeFitness(
  results['final']['accuracy'],
  results['final']['time']
)
results['final']['noFitness'] = computeFitness(
  results['final']['noAccuracy'],
  results['final']['time']
)
results['final']['yesFitness'] = computeFitness(
  results['final']['yesAccuracy'],
  results['final']['time']
)
cnnConfig = {
  'numFilters': 50,
  'kernelSize': 4,
  'numConv2DLayers': 1,
  'poolingSize': 2,
  'denseNodes': 20,
  'learningRate': 0.01,
}
# Print final config
print(f"Final config - \n\
      Image Size: {cnnConfig['imageSize']} \n\
      Filter kernel size: {cnnConfig['kernelSize']}x{cnnConfig['kernelSize']} \n\
      Filter quantity per Conv2D layer: {cnnConfig['numFilters']} \n\
      Number of Conv2D layers: {cnnConfig['numConv2DLayers']} \n\
      Pooling filter size: {cnnConfig['poolingSize']}x{cnnConfig['poolingSize']} \n\
      Dense layer node quantity: {cnnConfig['denseNodes']} \n\
      Optimizer learning rate: {cnnConfig['learningRate']}")
# Print final results
print(f"Final Results - \n\
      Accuracy: {results['final']['accuracy']} \n\
      'Yes' Accuracy: {results['final']['yesAccuracy']} \n\
      'No' Accuracy: {results['final']['noAccuracy']} \n\
      'Yes' Fitness: {results['final']['yesFitness']} \n\
      'No' Fitness: {results['final']['noFitness']} \n\
      Training Time: {results['final']['time']} \n\
      Fitness: {results['final']['fitness']}")