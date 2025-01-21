import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, Add, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def resnet_block(input_layer, filters, conv_size, reduce_size=False):
    if reduce_size:
        stride = 2
    else:
        stride = 1

    x = Conv2D(filters, conv_size, strides=stride, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, conv_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    if reduce_size:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same")(input_layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_layer

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)


    num_blocks_list = [2, 2, 2, 2]
    for i, num_blocks in enumerate(num_blocks_list):
        for j in range(num_blocks):
            filters = 64 * (2**i)
            if j == 0 and i != 0:
                x = resnet_block(x, filters, 3, reduce_size=True)
            else:
                x = resnet_block(x, filters, 3)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def _bytes_feature(value):
    """string / byte 타입을 받아서 byte_list로 변환"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

diseases = ['결막염', '무증상', '안검내반증', '유루증', '핵경화']
image_directories = ['/Users/gim-yeon-u/Desktop/안구/TL2/개/안구/일반/결막염/유', 
'/Users/gim-yeon-u/Desktop/안구/TL2/개/안구/일반/결막염/무',
 '/Users/gim-yeon-u/Desktop/안구/TL2/개/안구/일반/안검내반증/유', 
'/Users/gim-yeon-u/Desktop/안구/TL2/개/안구/일반/유루증/유', 
'/Users/gim-yeon-u/Desktop/안구/TL2/개/안구/일반/핵경화/유']
dataframe = pd.DataFrame()
dataframe = pd.DataFrame()

for index, image_directory in enumerate(image_directories):
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg')]
    temp_df = pd.DataFrame({
        'filename': image_files,
        'label': [diseases[index] for _ in image_files]
    })
    dataframe = pd.concat([dataframe, temp_df], ignore_index=True)

train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

def create_tfrecord(dataframe, tfrecord_filename):
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for filename, label in zip(dataframe['filename'], dataframe['label']):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.uint8)
            label = label_to_index[label]
            example = serialize_example(tf.io.encode_jpeg(image).numpy(), label)
            writer.write(example)

label_to_index = {name: index for index, name in enumerate(diseases)}

create_tfrecord(train_df, 'train_dataset.tfrecords')
create_tfrecord(test_df, 'test_dataset.tfrecords')

def _parse_function(proto):
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    label = tf.one_hot(parsed_features['label'], depth=len(diseases))
    return image, label

def load_dataset(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = load_dataset('train_dataset.tfrecords', 32)
test_dataset = load_dataset('test_dataset.tfrecords', 32)

model = build_resnet18(input_shape=(224, 224, 3), num_classes=len(diseases))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=5, validation_data=test_dataset)
