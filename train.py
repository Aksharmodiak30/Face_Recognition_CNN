from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np




TRAINING_DIR = "./train"
train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=4,
                                                    target_size=(150, 150))
VALIDATION_DIR = "./test"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                         batch_size=2,
                                                         target_size=(150, 150))



model =Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])





checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


history = model.fit_generator(train_generator,
                              epochs=15,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])


