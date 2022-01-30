# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:49:22 2022

@author: zacharywe
"""
import numpy as np 
import pandas as pd 
from scipy import stats as ss

#Our Data
df = pd.read_excel('rpg.xlsx')
alpha = .05
dof = 33
conf = .95

mean_wp = np.mean(df['Win Percent'])
stdev_wp = np.std(df['Win Percent'],ddof=1)
mean_orpg = np.mean(df['ORPG'])
stdev_orpg = np.std(df['ORPG'],ddof=1)

''' Mean of Win Percent: 0.61496, Mean of ORPG: 3.48823'''
''' STDEV of Win Percent: 0.19028, STDEV of ORPG: 0.38436 '''

''' 2-SAMPLE T-TEST '''

'''
H0: There is no difference between having a player with > 3 ORPG and Win Percent
HA: There IS a difference between having a player with > 3 ORPG and Win Percent
'''

df_winp = df['Win Percent']
df_orpg = df['ORPG']

two_samp_t = ss.ttest_ind(df_winp,df_orpg)
print('')
print(two_samp_t)
print('')
''' T-Stat -39.063887, P: 2.4e-47'''
print('Because P-Value < Alpha, we reject the null hypothesis. Having a player with > 3 ORPG affects Win Percent')

