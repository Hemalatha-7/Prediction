import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
dataset="data"
cloudy_dir=os.path.join(dataset,"cloudy")
desert_dir=os.path.join(dataset,"desert")
green_area_dir=os.path.join(dataset,"green_area")
water_dir=os.path.join(dataset,"water")
image_size=(64,64)
batch_size=32
epochs=10
datagen=ImageDataGenerator(rescale=1.0/255.0,
                           validation_split=0.2,
                           rotation_range=0.2,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           fill_mode='nearest'
                           )
train_generator=datagen.flow_from_directory(dataset,
                                             target_size=image_size,
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             subset='training',
                                             shuffle=True)
validation_generator=datagen.flow_from_directory(dataset,
                                                   target_size=image_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   subset='validation',
                                                   shuffle=True)
model=Sequential([Conv2D(32,(3,3),activation='relu',input_shape=(image_size[0],image_size[1],3)),
                MaxPooling2D((2,2)),
                Conv2D(64,(3,3),activation='relu'),
                MaxPooling2D((2,2)),
                Conv2D(128,(3,3),activation='relu'),
                MaxPooling2D((2,2)),
                Flatten(),
                Dense(128,activation='relu'),
                Dropout(0.5),
                Dense(4,activation='softmax')])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
                metrics=['accuracy'])
history=model.fit(train_generator,
                  validation_data=validation_generator,
                  epochs=epochs,
                  steps_per_epoch=train_generator.samples//batch_size,
                  validation_steps=validation_generator.samples//batch_size)
model.save('sic_model.h5')

