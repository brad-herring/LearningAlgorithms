from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf

from numpy import nan

CSV_COLUMN_NAMES = ['position', 'Height', 'Weight', 'FortyYard', 'TwentySS',
                    'ThreeCone', 'Vertical', 'Broad', 'Bench']

POSITIONS = {'RB': 1, 'WR': 2, 'OLB': 3, 'FS': 4, 'DE': 5, 'TE': 6, 'ILB': 7, 'DT': 8, 'P': 9, 'QB': 10, 'OG': 11,
             'OT': 12, 'K': 13, 'FB': 14, 'SS': 15, 'LS': 16, 'CB': 17, 'C': 18, 'NT': 19, 'OC': 20}

train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv',
                    names=CSV_COLUMN_NAMES)
test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\NFL Stuff\combine.csv',
                   names=CSV_COLUMN_NAMES)

train.position = [POSITIONS[item] for item in train.position]
test.position = [POSITIONS[item] for item in test.position]

train_target = train.pop('position')
test_target = test.pop('position')

train[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']] \
    = train[['Height', 'Weight', 'FortyYard', 'TwentySS', 'ThreeCone', 'Vertical', 'Broad', 'Bench']].replace(0, nan)

print(train.head())

