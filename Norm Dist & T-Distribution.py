# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 11:54:13 2022

@author: zacharywe
"""
import numpy as np 
import pandas as pd 
from scipy import stats as ss
from scipy.interpolate import BSpline, make_interp_spline
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
%matplotlib inline


#Normal Dist Setup
mu = 0; sigma = 1
x_norm = np.linspace(-np.e, np.e,100) # Very close to np.linspace(ss.norm.ppf(0.0033),ss.norm.ppf(0.9967), 100) 

y_norm_pdf = ss.norm.pdf(x_norm, loc=mu, scale=sigma)
y_norm_cdf = ss.norm.cdf(x_norm, loc=mu, scale=sigma)
y_norm_icdf= ss.norm.ppf(x_norm, loc=mu, scale=sigma)


#Norm Dist Plot
fig, axes = plt.subplots(1, 3, figsize=(14,4))
ax1, ax2, ax3 = axes

ax1.plot(x_norm, y_norm_pdf, 'r-', lw=2, alpha=0.5, label='PDF')
ax1.set_xlabel('x (in STDEVs)')
ax1.set_ylabel('Probability density')

ax2.plot(x_norm, y_norm_cdf, 'b-', lw=2, alpha=0.5, label='CDF')
ax2.set_xlabel('x (in STDEVs)')
ax2.set_ylabel('Probability')

ax3.plot(x_norm, y_norm_icdf, 'g-', lw=2, alpha=0.5, label='$CDF^{-1}$')
ax3.set_xlabel('Probability')
ax3.set_ylabel('Quantile')

for ax in axes:
    ax.set_title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
    ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()

#Student's T Distribution
fig, axes = plt.subplots(1, 3, figsize=(15,4))
ax1, ax2, ax3 = axes

ax1c = iter(plt.cm.autumn(np.linspace(0,1,4)[::-1]))
ax2c = iter(plt.cm.winter(np.linspace(0,1,4)[::-1]))
ax3c = iter(plt.cm.summer(np.linspace(0,1,4)[::-1]))

for df in [10,20,33,40]:
    x_t = np.linspace(ss.t.ppf(0.0033, df), ss.t.ppf(0.9967, df), 100) # Set close to norm for consistency

    ax1.plot(x_t, ss.t.pdf(x_t, df), c=next(ax1c), lw=2, alpha=0.6, label=f'PDF: {df} DoF')
    ax2.plot(x_t, ss.t.cdf(x_t, df), c=next(ax2c), lw=2, alpha=0.6, label=f'CDF: {df} DoF')
    ax3.plot(x_t, ss.t.ppf(x_t, df), c=next(ax3c), lw=2, alpha=0.6, label='$CDF^{-1}$: %d DoF' % (df))

ax1.plot(x_norm, y_norm_pdf, '-.',color='black', lw=2, alpha=0.6, label='PDF: norm')
ax1.update({'xlabel':'x (in SDs)', 'ylabel':'Probability density', 'title':"Student's T: Probability Density"})

ax2.plot(x_norm, y_norm_cdf, '-.',color='black', lw=2, alpha=0.6, label='CDF: norm')
ax2.update({'xlabel':'x (in SDs)', 'ylabel':'Probability', 'title':"Student's T: Cumulative Density"})

ax3.plot(x_norm, y_norm_icdf,'-.',color='black', lw=2, alpha=0.6, label='$CDF^{-1}$: norm')
ax3.update({'xlabel':'Probability', 'ylabel':'Quantile', 'title':"Student's T: Inverse CDF"})

[a.legend() for a in axes]
plt.tight_layout()
plt.show()

