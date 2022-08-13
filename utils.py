import matplotlib.pyplot as plt 
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import copy

def sgd(obj, x, lr):
    noise = np.random.randn()
    #print('noise=', noise)
    x = x - lr * grad(obj)(x) - lr * noise 
    #print('grad=', grad(obj)(x))
    return x

def averaged_sgd(obj, x, x_ave, lr, t):
    #x_ave と　x二つの更新を行う
    beta = 1/(t+1)
    noise = np.random.randn()
    x = x - lr * grad(obj)(x) - lr * noise
    x_ave = beta*t*x_ave + beta*x
    return x, x_ave

def noisy_sgd(obj, x, lr, rho):
    noise_1 = np.random.randn()
    noise_2 = np.random.randn()
    x = x - lr * grad(obj)(x - rho*noise_2) -lr * noise_1  
    return x

'''
def object_function(x):
    return x**2 + x * jnp.sin(0.5*x**3) + 1.5*jnp.sin(5*x)
'''

def optimize(configs, opt, obj):
    list = []
    for i in tqdm(range(configs.sample_num)):
        x = configs.init + 10 * np.random.randn()
        if 'Ave_SGD' in opt:
            x_ave = copy.deepcopy(x)
        for j in range(configs.iter_num):
            if opt == 'SGD_large':
                x = sgd(obj, x, configs.lr_large)
            elif opt == 'SGD_small':
                x = sgd(obj, x, configs.lr_small)
            elif opt == 'Ave_SGD_large':
                x, x_ave = averaged_sgd(obj, x, x_ave, configs.lr_large, j)
            elif opt == 'Ave_SGD_small':
                x, x_ave = averaged_sgd(obj, x, x_ave, configs.lr_small, j)
            elif opt == 'Noisy_SGD_large':
                x = noisy_sgd(obj, x, configs.lr_large, configs.rho)
            elif opt == 'Noisy_SGD_small':
                x = noisy_sgd(obj, x, configs.lr_small, configs.rho)
            else:
                print('opt is not define')
        if 'Ave_SGD' in opt:
            list.append(x_ave)
        else:
            list.append(x)
    arr = np.array(list)
    print('complete : ',opt)
    return arr

@dataclass
class configs():
    init: float = 5.0
    iter_num: int = 100
    sample_num: int = 10
    lr_large: float = 0.1
    lr_small: float = 0.00001
    rho: float = 1.0