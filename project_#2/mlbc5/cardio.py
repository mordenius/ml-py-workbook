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
mean_cardio_by_gender_and_chole = cardio_data.groupby(
    ['gender', 'cholesterol'])['cardio'].aggregate('mean').unstack()
# print(mean_cardio_by_gender_and_chole)

# Append column `smoke_and_alco`
cardio_data['smoke_and_alco'] = cardio_data[[
    'smoke', 'alco']].apply(lambda x: x[0] * x[1], axis=1)
mean_cardio_by_smoke_and_alco = cardio_data.groupby(
    ['smoke_and_alco'])['cardio'].aggregate('mean')
# print(mean_cardio_by_smoke_and_alco)

# Draw plot for `height` and `weight`
# plt.figure(figsize=(12,8))
# plt.scatter(cardio_data.weight, cardio_data.height, s=9, c=cardio_data.cardio, cmap='seismic')
# plt.colorbar()
# plt.xlabel('weight')
# plt.ylabel('height')
# plt.show()

# Draw plot for `ap_lo` and `ap_hi`
# plt.figure(figsize=(12,8))
# plt.scatter(cardio_data.ap_lo, cardio_data.ap_hi, s=9, c=cardio_data.cardio, cmap='seismic')
# plt.colorbar()
# plt.xlabel('ap_lo')
# plt.ylabel('ap_hi')
# plt.show()

# f, axes = plt.subplots(6, 1)
# sns.boxplot(cardio_data.weight, ax=axes[0])
# sns.distplot(cardio_data.weight, ax=axes[1])
# sns.boxplot(cardio_data.height, ax=axes[2])
# sns.distplot(cardio_data.height, ax=axes[3])
# sns.boxplot(cardio_data.age, ax=axes[4])
# sns.distplot(cardio_data.age, ax=axes[5])
# plt.show()

# Data cleansing
mask = (cardio_data.ap_hi == 1) & (cardio_data.ap_lo > 100)

cardio_data.loc[mask, 'ap_hi'] = (
    cardio_data.loc[mask, 'ap_hi'] * 100) + (cardio_data.loc[mask, 'ap_lo'] // 100)
cardio_data.loc[mask, 'ap_lo'] = (
    cardio_data.loc[mask, 'ap_lo'].mod(100)//10)*10
cardio_data.loc[cardio_data.ap_hi == 14900, 'ap_hi'] = 140
cardio_data.loc[cardio_data.ap_hi > 4000, 'ap_hi'] = 100
mask = (cardio_data.ap_lo == 0) & (cardio_data.ap_hi >= 300)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = cardio_data.loc[mask,
                                                                                 'ap_hi'] // 10, cardio_data.loc[mask, 'ap_hi'].mod(10)*10
cardio_data.loc[cardio_data.ap_hi.isin([400, 401]), 'ap_hi'] = 140
cardio_data.loc[cardio_data.ap_hi == 701, 'ap_hi'] = 170
cardio_data.loc[cardio_data.ap_hi == 1420, 'ap_hi'] = 140
cardio_data.loc[cardio_data.ap_hi == 1620, 'ap_hi'] = 160
cardio_data.loc[cardio_data.ap_hi.isin([1130, 1110]), 'ap_hi'] = 110
cardio_data.loc[cardio_data.ap_hi == 960, 'ap_hi'] = 90
cardio_data.loc[cardio_data.ap_hi >= 300, 'ap_hi'] //= 10

cardio_data.loc[cardio_data.ap_lo == 4100, 'ap_lo'] = 140
cardio_data.loc[cardio_data.ap_lo == 10000, 'ap_lo'] = 100
cardio_data.loc[cardio_data.ap_lo == 5700, 'ap_lo'] = 75
cardio_data.loc[cardio_data.ap_lo == 6800, 'ap_lo'] = 80
cardio_data.loc[cardio_data.ap_lo == 4700, 'ap_lo'] = 70
cardio_data.loc[cardio_data.ap_lo == 1100, 'ap_lo'] = 110

cardio_data.loc[cardio_data.ap_lo >= 4000, 'ap_lo'] = (
    cardio_data.loc[cardio_data.ap_lo >= 4000, 'ap_lo']//1000)*10
cardio_data.loc[cardio_data.ap_lo == 1900, 'ap_lo'] = 90
cardio_data.loc[cardio_data.ap_lo == 1211, 'ap_lo'] = 120
cardio_data.loc[cardio_data.ap_lo >= 1200, 'ap_lo'] //= 10
cardio_data.loc[cardio_data.ap_lo >= 1000, 'ap_lo'] = 100
cardio_data.loc[cardio_data.ap_lo == 570, 'ap_lo'] = 75
cardio_data.loc[cardio_data.ap_lo.isin([850, 585]), 'ap_lo'] = 85
cardio_data.loc[cardio_data.ap_lo >= 300, 'ap_lo'] = (
    cardio_data.loc[cardio_data.ap_lo >= 300, 'ap_lo']//100)*10

cardio_data.loc[cardio_data.ap_hi < 0, 'ap_hi'] *= -1
cardio_data.loc[cardio_data.ap_lo < 0, 'ap_lo'] *= -1

mask = (cardio_data.ap_hi == 1) & (cardio_data.ap_lo <= 100)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask,
                                                'ap_lo'] = cardio_data.loc[mask, 'ap_hi'] * 100 + cardio_data.loc[mask, 'ap_lo'], 80

mask = (cardio_data.ap_hi == 10) & (cardio_data.ap_lo.isin([0, 160]))
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 100, 60
mask = (cardio_data.ap_hi == 11) & (cardio_data.ap_lo == 120)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 120, 80
mask = (cardio_data.ap_hi == 11) & (cardio_data.ap_lo == 57)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 115, 70
mask = (cardio_data.ap_hi == 12) & (cardio_data.ap_lo.isin([0, 140]))
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 120, 80
mask = (cardio_data.ap_hi == 13) & (cardio_data.ap_lo == 58)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 135, 80
mask = (cardio_data.ap_hi == 13) & (cardio_data.ap_lo == 0)
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = 130, 80

cardio_data.loc[cardio_data.ap_hi.between(10, 24), 'ap_hi'] *= 10
cardio_data.loc[cardio_data.ap_hi == 7, 'ap_hi'] - 120

mask = (cardio_data.ap_lo == 0) & (cardio_data.ap_hi == 108)
cardio_data.loc[mask, 'ap_lo'], cardio_data.loc[mask, 'ap_hi'] = 100, 80
mask = (cardio_data.ap_lo == 0) & (cardio_data.ap_hi == 118)
cardio_data.loc[mask, 'ap_lo'], cardio_data.loc[mask, 'ap_hi'] = 110, 80
mask = (cardio_data.ap_hi == 12) & (cardio_data.ap_lo.isin([90, 80]))
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask,
                                                'ap_lo'] = 120, cardio_data.loc[mask, 'ap_hi']
mask = (cardio_data.ap_hi == 12) & (
    cardio_data.ap_lo.isin([138, 117, 149, 148]))
cardio_data.loc[mask, 'ap_hi'], cardio_data.loc[mask, 'ap_lo'] = (
    cardio_data.loc[mask, 'ap_hi']//10)*10, cardio_data.loc[mask, 'ap_hi'].mod(10)*10
mask = (cardio_data.ap_lo == 0) & (cardio_data.ap_hi == 50)
cardio_data.loc[mask, 'ap_lo'], cardio_data.loc[mask, 'ap_hi'] = 150, 100
cardio_data.loc[cardio_data.ap_lo == 0, 'ap_lo'] = 80


cardio_data.loc[(cardio_data.height == 169) & (
    cardio_data.weight == 16.3), 'weight'] = 63
cardio_data.loc[cardio_data.weight.between(10, 15), 'weight'] *= 10
cardio_data.loc[cardio_data.weight.between(20, 29), 'weight'] += 100
cardio_data.loc[(cardio_data.height == 170) & (
    cardio_data.weight == 31) & (cardio_data.gender == 1), 'weight'] += 100
cardio_data.loc[(cardio_data.height == 179) & (
    cardio_data.weight == 165), 'weight'] = 65
cardio_data.loc[(cardio_data.height == 186) & (
    cardio_data.weight == 200), 'weight'] = 100
mask = (cardio_data.weight.between(100, 200)) & (cardio_data.height < 100)
cardio_data.loc[mask, 'weight'], cardio_data.loc[mask,
                                                 'height'] = cardio_data.loc[mask, 'height'], cardio_data.loc[mask, 'weight']
cardio_data.loc[cardio_data.height > 240, 'height'] -= 100
cardio_data.loc[cardio_data.height < 80, 'height'] += 100
cardio_data.loc[(cardio_data.height == 119) & (
    cardio_data.weight == 155), 'weight'] = 55
cardio_data.loc[(cardio_data.height == 128) & (
    cardio_data.weight == 128), 'weight'] = 76
cardio_data.loc[(cardio_data.height < 150) & (
    cardio_data.weight >= 130), 'weight'] -= 100
cardio_data.loc[(cardio_data.height > 140) & (
    cardio_data.weight >= 100), 'weight'] -= 100
cardio_data.loc[(cardio_data.weight == cardio_data.height)
                & (cardio_data.gender == 2), 'weight'] = 70
cardio_data.loc[(cardio_data.weight == cardio_data.height)
                & (cardio_data.gender == 1), 'weight'] = 75

cardio_data.loc[(cardio_data.weight.between(160, 199)) & (
    cardio_data.ap_hi <= 120) & (cardio_data.gender == 1), 'weight'] -= 100
cardio_data.loc[(cardio_data.weight.between(140, 199)) & (cardio_data.ap_hi <= 120) & (
    cardio_data.gender == 12) & (cardio_data.height < 175), 'weight'] -= 100

cardio_data.loc[cardio_data.height == 168 & (
    cardio_data.weight == 174), 'weight'] = 68

print(cardio_data.describe())
