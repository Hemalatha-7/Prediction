from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model=load_model('sic_model.h5')
img_path='data/green_area/Forest_2.jpg'
img=image.load_img(img_path,target_size=(64,64))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)/255.0
prediction=model.predict(img_array)
predicted_class=np.argmax(prediction,axis=1)
print("Predicted class:", predicted_class)
