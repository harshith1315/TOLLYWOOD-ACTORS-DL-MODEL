from django.shortcuts import render
import cv2
import h5py
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

def hello(request):
    return render(request,'test.html')
def index(request):
    if request.method == 'GET':
        ypred=''
        f=request.GET['photo']
        print(f)
        t = cv2.imread(f)
        with h5py.File('my_model_compressed.h5', 'r') as file:
            loaded_model_json = file['model'][()]
            loaded_model = model_from_json(loaded_model_json)
        testimg = cv2.resize(t, (256, 256))
        testimg = img_to_array(testimg) / 255
        h = np.expand_dims(testimg, axis=0)
        r = loaded_model.predict(h)
        classname = ['akhil', 'brahmi', 'nani', 'rc']
        ypred = classname[np.argmax(r)]
        print(ypred)
    return render(request,'test.html',{'output': ypred})
