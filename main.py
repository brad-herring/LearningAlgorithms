from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output

import tensorflow as tf

CSV_COLUMN_NAMES = ['position', 'heightinchestotal', 'weight', 'fortyyd', 'twentyss', 'threecone', 'vertical', 'broad', 'bench']
POSITION = ['RB', 'WR', 'OLB', 'FS', 'DE', 'TE', 'ILB', 'DT', 'P', 'QB', 'OG', 'OT', 'K', 'FB', 'SS', 'LS', 'CB', 'C', 'NT', 'OC']

train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv')
test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv')

train_y = train.pop('position')
test_y = test.pop('position')


def input_fn(features, labels, training=True, batch_size=256):
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key,))
print(my_feature_columns)

# DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))






