from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os


for _ in range(1):

    model_path = f'{os.getcwd()}\\model\\model_weights.h5'
    model = load_model(model_path)

    img_path = f'{os.getcwd()}\\model\\very-cute-dog.jpg'
    img = image.load_img(img_path, target_size=(406, 612))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 

    
    prediction = model.predict(img_array)
    print(prediction)
    if prediction[0][0] >= 0.5:
        print("The model classified the image as a dog.")
    else:
        print("The model classified the image as a bagel.")
