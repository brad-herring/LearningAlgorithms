from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from numpy import nan

CSV_COLUMN_NAMES = ['position', 'Height', 'Weight', 'FortyYard', 'TwentySS',
                    'ThreeCone', 'Vertical', 'Broad', 'Bench']

POSITIONS = {'RB': 1, 'WR': 2, 'OLB': 3, 'FS': 4, 'DE': 5, 'TE': 6, 'ILB': 7, 'DT': 8, 'P': 9, 'QB': 10, 'OG': 11,
             'OT': 12, 'K': 13, 'FB': 14, 'SS': 15, 'LS': 16, 'CB': 17, 'C': 18, 'NT': 19, 'OC': 20}

POSITIONS_REV = {1: 'RB', 2: 'WR', 3: 'OLB', 4: 'FS', 5: 'DE',6: 'TE', 7: 'ILB', 8: 'DT', 9: 'P', 10: 'QB', 11: 'OG',
                 12: 'OT', 13: 'K', 14: 'FB', 15: 'SS', 16: 'LS', 17: 'CB', 18: 'C', 19: 'NT', 20: 'OC'}

# Create training and testing datasets
train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv',
                    names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv',
                   names=CSV_COLUMN_NAMES, header=0)

# Replaces zero values with NaN (missing value) from numpy
train[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']] \
    = train[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']].replace(0, nan)

# Eliminates rows with missing values
train.dropna(inplace=True)

# Replaces zero values with NaN (missing value) from numpy
test[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']] \
    = test[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']].replace(0, nan)

# Eliminates rows with missing values
test.dropna(inplace=True)

# Replaces position abbreviation with numerical value
train.position = [POSITIONS[item] for item in train.position]
test.position = [POSITIONS[item] for item in test.position]

# Separates target information from the data
train_target = train.pop('position')
test_target = test.pop('position')

# Standardization of training and testing data
names = train.columns

scaler = preprocessing.StandardScaler()

train = scaler.fit_transform(train)
train = pd.DataFrame(train, columns=names)
test = scaler.transform(test)
test = pd.DataFrame(test, columns=names)


# Create input function
def input_fn(features, labels, training=True, batch_size=256):
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)

# Feature columns describe how to use the input
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 210 and 70 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[210, 70],
    n_classes=21)

# Train the model
classifier.train(
    input_fn=lambda: input_fn(train, train_target, training=True),
    steps=10000)

# Evaluate the accuracy of the model
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_target, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# Creates input function without labels
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


# Gathers user input to predict the position
features = ['Height', 'Weight', 'FortyYard', 'TwentySS',
            'ThreeCone', 'Vertical', 'Broad', 'Bench']

predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False
    predict[feature] = [float(val)]
    valid = False

print(predict)

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        POSITIONS_REV[class_id], 100 * probability))
