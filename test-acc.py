import pandas as pd
import os
import tensorflow as tf
import pandas as pd


# Develop dictionary mapping from classes to labels
classes = pd.read_csv('./archive/meta/meta/classes.txt', header=None, index_col=0,)
labels = pd.read_csv('./archive/meta/meta/labels.txt', header=None)
classes['map'] = labels[0].values
classes_to_labels_map = classes['map'].to_dict()
test_df = pd.read_csv('./archive/meta/meta/test.txt', header=None).apply(lambda x : './archive/images/' + x + '.jpg')


import json
with open('./classes_to_indice.json', 'rt') as f:
    classes_to_indice = json.load(f)


image_keys = []
image_values = []
for path in test_df[0].tolist():
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

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [224, 224])
    
    return image/255.0,  tf.one_hot(table.lookup(file_path), 101)


testdata = tf.data.Dataset.from_tensor_slices(test_df[0].tolist())
batched = testdata.map(process_path,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=300).batch(32).prefetch(3)





model = tf.keras.models.load_model('mobilenet-frozen-shuffled-a1.h5')
model.compile(metrics=['accuracy', tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(),
                       tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.CategoricalCrossentropy()

            ])

print( model.evaluate(batched) )