import cv2
import numpy as np
from keras.models import load_model
model=load_model("./model2-021.model")


labels_dict={0:'aksh',1:'jan'}

face_img = cv2.imread("janu.jpg", 1)
resized=cv2.resize(face_img,(150,150))
normalized=resized/255.0
reshaped=np.reshape(normalized,(1,150,150,3))
reshaped = np.vstack([reshaped])
result=model.predict(reshaped)
print(result)

label=np.argmax(result,axis=1)[0]

print(labels_dict[label])
