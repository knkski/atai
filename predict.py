import numpy as np
import sys
import keras
from scipy.ndimage import imread

# Load model
model = keras.models.load_model('model.h5')

# Convert image file to input readable by the model
image = imread(sys.argv[1]).reshape(1, 28, 28, 1)

# Run the model against the input, and convert from one-hot binary output into
# a human-friendly character prediction
prediction = chr(ord('A') + np.argmax(model.predict(image)))

print('The predicted letter for this input is %s' % prediction)
