import cv2
import h5py
from cv2 import imshow
from tensorflow.keras.models import model_from_json

with h5py.File('my_model_compressed.h5', 'r') as file:
    loaded_model_json = file['model'][()]
    loaded_model = model_from_json(loaded_model_json)

t = cv2.imread('89798774.jpg')

# Display the original image
cv2.imshow('Original Image', t)
cv2.waitKey(0)
cv2.destroyAllWindows()

testimg = cv2.resize(t, (256, 256))
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

testimg = img_to_array(testimg) / 255
h = np.expand_dims(testimg, axis=0)
r = loaded_model.predict(h)
classname = ['akhil', 'brahmi', 'nani', 'rc']
ypred = classname[np.argmax(r)]
print(ypred)

