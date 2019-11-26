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

# Draw plot for `height` and `weight` 
plt.figure(figsize=(12,8))
plt.scatter(cardio_data.weight, cardio_data.height, s=9, c=cardio_data.cardio, cmap='seismic')
plt.colorbar()
plt.xlabel('weight')
plt.ylabel('height')
# plt.show()

# Draw plot for `ap_lo` and `ap_hi` 
plt.figure(figsize=(12,8))
plt.scatter(cardio_data.ap_lo, cardio_data.ap_hi, s=9, c=cardio_data.cardio, cmap='seismic')
plt.colorbar()
plt.xlabel('ap_lo')
plt.ylabel('ap_hi')
# plt.show()

f, axes = plt.subplots(6, 1)

sns.boxplot(cardio_data.weight, ax=axes[0])
sns.distplot(cardio_data.weight, ax=axes[1])
sns.boxplot(cardio_data.height, ax=axes[2])
sns.distplot(cardio_data.height, ax=axes[3])
sns.boxplot(cardio_data.age, ax=axes[4])
sns.distplot(cardio_data.age, ax=axes[5])
plt.show()
