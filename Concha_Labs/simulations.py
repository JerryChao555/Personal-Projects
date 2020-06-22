import csv
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile

# Version 4/7: q2 only supports two curves and mse. noise supports normal and uniform.

# Optimization: create a bunch of noise at the same time and retrieve only when needed

rand_cache = []
unif_cache = []
MSE_CONSTANT = 1/3
COV_CONSTANT = 0.5

def noise(low, high, d='normal', cache=100):
    if d == 'normal':
        if len(rand_cache) == 0:
            temp_cache = np.random.randn(cache)
            # Throw away if num > 3 or < -3 (occurs less than 1% of the time)
            temp_cache = temp_cache[(temp_cache < 3) & (temp_cache > -3)]
            temp_cache = temp_cache / 6 + 0.5
            rand_cache.extend(temp_cache * (high - low) + low)
        return rand_cache.pop()
    elif d == 'uniform':
        if len(unif_cache) == 0:
            unif_cache.extend(np.random.uniform(low, high, cache))
        return unif_cache.pop() 
    # Could further implement other distribution
    elif d == '':
        return 0
    else:
        raise NotImplementedError

# Take in a given frequency, data and volumes we can play. volumes MUST be in INCREASING order
def q1(data, freq, volumes, noise_max=5, noise_min=0, noise_type='normal'):
    x = data[freq]
    #x = np.random.choice(x) + noise(x.min()*0.1, x.max()*0.1)
    x = x + noise(noise_min, noise_max, noise_type)
    for i in np.arange(len(volumes)):
        if x < volumes[i]:
            return volumes[i]
    return volumes[-1]

def cov_calc(nums):
    diff = nums[:,0,0] - nums[:,0, 1]
    temp = (diff > 0) * COV_CONSTANT + COV_CONSTANT
    return (diff * temp) ** 2

def mse_calc(original, curves):
    curves = np.array([curve for curve in curves if len(curve) == len(original)])
    
    a, b = 1 - MSE_CONSTANT, MSE_CONSTANT
    # matrix operations
    diff = (original - curves)
    temp = (diff > 0) * a + b
    return np.mean((diff * temp) ** 2, axis=1)

# Returns a number in [0, len(curves)+1] for answer. 
# len(curves) represent "equally clear", and len(curves)+1 represent "equally unclear"
def q2(data, curves, similar_thres=1, unclear_thres=20, mse_cov_weight=[1, 0], noise_max=5, noise_min=0, noise_type='normal', cache=100):
    
    x = data
    assert(len(data) == len(curves[0]))

    #error = noise(x.min()*0.1, x.max()*0.1)
    error = noise(noise_min, noise_max, d=noise_type, cache=cache)

    mse_scores = mse_calc(x+error, curves)
    temp = np.array([np.cov(x+error, curve) for curve in curves])
    cov_scores = cov_calc(temp)
    scores = mse_cov_weight[0] * mse_scores + mse_cov_weight[1] * cov_scores 
    
    # unclear_thres = np.mean(x**2) 
    # similar_thres = unclear_thres * 0.1
    
    if np.all(scores > unclear_thres):
        # Both unclear
        return len(curves)+1
    elif abs(scores[0] - scores[1]) < similar_thres:
        # Both clear
        return len(curves)
    else:
        return np.argmin(scores)


def noise_many(num, low, high, d='normal'):
    if d == 'normal':
        new_num = round(num * 1.1)
        randn = np.random.randn(new_num)
        # Throw away if num > 3 or < -3 (occurs less than 1% of the time)
        randn = randn[(randn < 3) & (randn > -3)]
        return randn[:num] / 6 + 0.5
    elif d == 'uniform':
        return np.random.uniform(low, high, num)
    else:
        raise NotImplementedError


"""
nums has size batch_size * curve_size * 2 * 2
output should be batch_size * 2
"""
def cov_calc_many(nums):
    diff = nums[:,:,0,0] - nums[:,:,0,1]
    temp = (diff > 0) * COV_CONSTANT + COV_CONSTANT
    return (diff * temp) ** 2

"""
original is a matrix of size batch_size * 7

output should be batch_size * 2
"""
def mse_calc_many(original, curves):
    curves = np.array([curve for curve in curves if len(curve) == len(original[0])])
    
    a, b = 1 - MSE_CONSTANT, MSE_CONSTANT
    # matrix operations
    # diff will have shape batch_size * 2 * 7
    diff = np.apply_along_axis(np.subtract, 1, original, curves)
    temp = (diff > 0) * a + b
    return np.mean((diff * temp) ** 2, axis=2)

"""
data should have shape batch_size * 7
"""
def q1_many(data, freq, volumes, noise_max=5, noise_min=0, noise_type='normal'):
    x = data[:, freq]
    num = data.shape[0]

    x = x + noise_many(num, noise_min, noise_max, d=noise_type)

    ret_vol = []
    flag = False

    for j in range(num):
        y = x[j]
        for i in np.arange(len(volumes)):
            if y < volumes[i]:
                ret_vol.append(volumes[i])
                flag = True
                break

        if not flag:
            ret_vol.append(volumes[-1])
        else:
            flag = False
    
    return np.array(ret_vol)

"""
data should have shape batch_size * 7
"""
def q2_many(data, curves, similar_thres=1, unclear_thres=20, mse_cov_weight=[1, 0], noise_max=5, noise_min=0, noise_type='normal'):
    x = data
    num = x.shape[0]

    error = noise_many(num, noise_min, noise_max, d=noise_type)

    x = x + error.reshape(-1, 1)

    mse_scores = mse_calc_many(x, curves)

    if mse_cov_weight[1] == 0:
        cov_scores = np.zeros(mse_scores.shape)
    else:
        temp = [np.apply_along_axis(np.cov, 1, x, curve) for curve in curves]
        temp = np.array(temp)
        cov_scores = cov_calc_new(temp)

    scores = mse_cov_weight[0] * mse_scores + mse_cov_weight[1] * cov_scores 
    
    # unclear_thres = np.mean(x**2) 
    # similar_thres = unclear_thres * 0.1
    results = []
    len_curv = len(curves)

    for i in range(num):
        score = scores[i]
        if np.all(score > unclear_thres):
            # Both unclear
            results.append(len_curv+1)
        elif abs(score[0] - score[1]) < similar_thres:
            # Both clear
            results.append(len_curv)
        else:
            results.append(np.argmin(score))
    
    return np.array(results)