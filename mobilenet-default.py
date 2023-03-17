import os
import pandas as pd
import numpy as np
import os
import PIL.Image
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import json

# Develop dictionary mapping from classes to labels
classes = pd.read_csv('./archive/meta/meta/classes.txt', header=None, index_col=0,)
labels = pd.read_csv('./archive/meta/meta/labels.txt', header=None)
classes['map'] = labels[0].values
classes_to_labels_map = classes['map'].to_dict()
train_df = pd.read_csv('./archive/meta/meta/train.txt', header=None).apply(lambda x : './archive/images/' + x + '.jpg')
# shuffle entire dataset
train_df = shuffle(train_df)

with open('./classes_to_indice.json', 'rt') as f:
    classes_to_indice = json.load(f)


image_keys = []
image_values = []
for path in train_df[0].tolist():
    image_keys.append(path)
    classid = path.split('/')[-2]
    image_values.append(classes_to_indice[classid])
table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(image_keys),
        values=tf.constant(image_values),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)

train = tf.data.Dataset.from_tensor_slices(train_df[0].tolist())

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [224, 224])

    return image/255.0,  tf.one_hot(table.lookup(file_path), 101)

train = train.map(process_path,num_parallel_calls=tf.data.experimental.AUTOTUNE)
batched = train.batch(32).prefetch(3)

# create MobileNet model with random initial weights and without top layers
# base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model = MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(101, activation='softmax')(x)

# combine base model and custom top layers into a single model
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    batched,
    epochs=6,
)

model.save(f'mobilenet-unfrozen.h5')