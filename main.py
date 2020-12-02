from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output

import tensorflow as tf

CSV_COLUMN_NAMES = ['position', 'heightinchestotal', 'weight', 'fortyyd', 'twentyss', 'threecone', 'vertical', 'broad', 'bench']
POSITIONS = {'RB': 1, 'WR': 2, 'OLB': 3, 'FS': 4, 'DE': 5, 'TE': 6, 'ILB': 7, 'DT': 8, 'P': 9, 'QB': 10, 'OG': 11,
             'OT': 12, 'K': 13, 'FB': 14, 'SS': 15, 'LS': 16, 'CB': 17, 'C': 18, 'NT': 19, 'OC': 20}

train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv', header=0)
test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv', header=0)

train.position = [POSITIONS[item] for item in train.position]
test.position = [POSITIONS[item] for item in test.position]

train_y = train.pop('position')
test_y = test.pop('position')


def input_fn(features, labels, training=True, batch_size=256):
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[210, 70],
    n_classes=21)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=10000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))