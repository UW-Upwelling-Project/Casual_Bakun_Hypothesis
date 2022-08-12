import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from patsy import cr

def upwell(ektrx, ektry, coast_angle):
  pi = 3.1415927
  degtorad = pi/180.
  alpha = (360 - coast_angle) * degtorad
  s1 = np.cos(alpha)
  t1 = np.sin(alpha)
  s2 = -1 * t1
  t2 = s1
  perp = (s1 * ektrx) + (t1 * ektry)
  para = (s2 * ektrx) + (t2 * ektry)
  return(perp/10)

def normalize(data):
    return (data - np.mean(data)) / np.std(data)

def detrend_differencing(data_list):
    new_data_list = []
    for i in range(len(data_list)):
        if i != 0:
            new_data_list.append(data_list[i] - data_list[i-1])
    return new_data_list

def detrend_linear_reg(data_list):
    X = [i for i in range(0, len(data_list))]
    X = np.reshape(X, (len(X), 1))
    y = data_list
    model = LinearRegression()
    model.fit(X, y)

    print("global trend: ", model.coef_)
    # calculate trend
    trend = model.predict(X)
    detrended = [y[i]-trend[i] for i in range(0, len(trend))]
    
    return detrended

def plot_smoothed(x, y, df=5):

    # Generate spline basis with different degrees of freedom
    x_basis = cr(x, df=df, constraints="center")

    # Fit model to the data
    model = LinearRegression().fit(x_basis, y)

    # Get estimates
    y_hat = model.predict(x_basis)

    return y_hat

def wavelet_transform_pipeline(fig, ax, upwelling_data, wavelet_hat, scale, freq, title):
    x_list = []

    for i in range(len(upwelling_data)):
        x_list.append(i)

    time = [x/12+1981 for x in x_list]

    scales = np.arange(1, scale)

    [wc, freq] = pywt.cwt(upwelling_data, scales, wavelet_hat, freq)

    power = wc**2
    period = 1. / freq

    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    contourlevels = np.log2(levels)
        
    # fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap='seismic')
        
    ax.set_title(title, fontsize=20)    
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel('Period (Years)', fontsize=18)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
        
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    # plt.show()