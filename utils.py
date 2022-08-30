import matplotlib.pyplot as plt 
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import copy

level = 1.2

def sgd(obj, x, lr):
    noise = level * np.random.uniform(-1,1)
    x -= lr * grad(x) + lr * noise 
    return x

def averaged_sgd(obj, x, x_ave, lr, t):
    beta = 1/(t+1)
    noise = level * np.random.uniform(-1,1)
    x -= lr * grad(x) + lr * noise
    x_ave *= beta*t
    x_ave += beta*x
    return x, x_ave

def noisy_sgd(obj, x, lr, rho):
    noise_1 = level * np.random.uniform(-1,1)
    noise_2 = level * np.random.uniform(-1,1)
    x -= lr * grad(x - rho*noise_2) + lr * noise_1  
    return x

def objective(x):
    #return x**2 + np.exp(-x**4/1000)*(x * np.sin(0.5*x**3) + 1.5*np.sin(5*x)) + 2
    return (1-np.exp(-x**8))*(x**2 + np.exp(-x**4/1000)*(x * np.sin(0.5*x**3) + 1.5*np.sin(5*x)) + 2)

def grad(x):
    #return 2 * x - np.exp(-x**4/1000)*(x**3/250)*(x * np.sin(0.5*x**3) + 1.5*np.sin(5*x)) + np.exp(-x**4/1000)*(np.sin(0.5*x**3) + 1.5*(x**3)*np.cos(0.5*x**3) + 7.5*np.cos(5*x)) 
    return 8*x**7*np.exp(-x**8)*(x**2 + np.exp(-x**4/1000)*(x * np.sin(0.5*x**3) + 1.5*np.sin(5*x)) + 2)+(1-np.exp(-x**8))*(2 * x - np.exp(-x**4/1000)*(x**3/250)*(x * np.sin(0.5*x**3) + 1.5*np.sin(5*x)) + np.exp(-x**4/1000)*(np.sin(0.5*x**3) + 1.5*(x**3)*np.cos(0.5*x**3) + 7.5*np.cos(5*x)))

def smoothed_objective(x, eta):
    y = 0
    num = 1000
    for i in range(num):
        y += objective(x+eta * level * np.random.uniform(-1,1)) / num
    return y

def optimize(configs, opt, obj):
    sol_list = []
    for i in tqdm(range(configs.sample_num)):
        x = configs.init + 10 * np.random.randn()

        if 'Ave_SGD' in opt:
            x_ave = copy.deepcopy(x)

        if opt == 'SGD_large':            
            for j in range(configs.iter_num):            
                x = sgd(obj, x, configs.lr_large)
        elif opt == 'SGD_small':
            for j in range(configs.iter_num):            
                x = sgd(obj, x, configs.lr_small)
        elif opt == 'Ave_SGD_large':
            for j in range(configs.iter_num):            
                x, x_ave = averaged_sgd(obj, x, x_ave, configs.lr_large, j)
        elif opt == 'Ave_SGD_small':
            for j in range(configs.iter_num):            
                x, x_ave = averaged_sgd(obj, x, x_ave, configs.lr_small, j)
        elif opt == 'Noisy_SGD_large':
            for j in range(configs.iter_num):            
                x = noisy_sgd(obj, x, configs.lr_large, configs.rho)
        elif opt == 'Noisy_SGD_small':
            for j in range(configs.iter_num):            
                x = noisy_sgd(obj, x, configs.lr_small, configs.rho)
        else:
            print('opt is not define')
            
        if 'Ave_SGD' in opt:
            sol_list.append(x_ave)
        else:
            sol_list.append(x)
    sol_arr = np.array(sol_list)
    print('complete : ',opt)
    return sol_arr

@dataclass
class config():
    init: float = 5.0
    iter_num: int = 100
    sample_num: int = 10
    lr_large: float = 0.1
    lr_small: float = 0.00001
    rho: float = 1.0