import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


dir_path = os.path.dirname(os.path.realpath(__file__))

# Reading Data
cardio_data = pd.read_csv(os.path.join(dir_path, 'cardio_train.csv'), sep=';')

# Remove column `id`
cardio_data.drop('id', axis=1, inplace=True)
# print(cardio_data.info())
# Not evenly: ~ 65/35 (%)

# Count in groups by gender
cnt_groupby_gender = cardio_data['gender'].value_counts()
# print(cnt_groupby_gender)

# Summary statistics
# print(cardio_data.describe())
# Abnormal values in columns [`ap_hi`, `ap_lo`] found

#
mean_cardio_by_gender_and_chole = cardio_data.groupby(['gender', 'cholesterol'])['cardio'].aggregate('mean').unstack()
# print(mean_cardio_by_gender_and_chole)

# Append column `smoke_and_alco`
cardio_data['smoke_and_alco'] = cardio_data[['smoke', 'alco']].apply(lambda x: x[0] * x[1], axis=1)
mean_cardio_by_smoke_and_alco = cardio_data.groupby(['smoke_and_alco'])['cardio'].aggregate('mean')
# print(mean_cardio_by_smoke_and_alco)
