import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np

diseases = ['각막궤양', '각막부골편', '결막염', '무증상']

image_directories = ['/Users/gim-yeon-u/Desktop/안구/TL2/고양이/안구/일반/각막궤양/유', '/Users/gim-yeon-u/Desktop/안구/TL2/고양이/안구/일반/각막부골편/유', '/Users/gim-yeon-u/Desktop/안구/TL2/고양이/안구/일반/결막염/유',  '/Users/gim-yeon-u/Desktop/안구/TL2/고양이/안구/일반/각막궤양/무']
dataframe = pd.DataFrame()

for index, image_directory in enumerate(image_directories):
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))][:2000]
    temp_df = pd.DataFrame({
        'filename': image_files,
        'label': [diseases[index] for _ in image_files]
    })
    dataframe = pd.concat([dataframe, temp_df], ignore_index=True)

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(diseases), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=30, validation_data=test_generator)

model.save('/Users/gim-yeon-u/Desktop/model_cateye')