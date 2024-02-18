from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os


for _ in range(1):
    # Load the saved model
    model_path = f'{os.getcwd()}\\model\\model_weights.h5'
    model = load_model(model_path)

    # Load the image
    img_path = f'{os.getcwd()}\\model\\very-cute-dog.jpg'
    img = image.load_img(img_path, target_size=(406, 612))  # Resize to match model input shape

    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Perform inference using the loaded model
    prediction = model.predict(img_array)
    print(prediction)
    # Interpret the prediction
    if prediction[0][0] >= 0.5:
        print("The model classified the image as a dog.")
    else:
        print("The model classified the image as a bagel.")
