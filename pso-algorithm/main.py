import numpy as np
from modules.test import Test

#########################################################################################
#################################### Test Functions #####################################
#########################################################################################

### sphere function ###
def function_0(x,y):
    return x**2+y**2         
#function from https://cienciadedatos.net/documentos/py02_optimizacion_pso

### Function ###
def function_1(x_0,x_1):
    #Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
    #múltiples mínimos locales y un único minimo global que se encuentra en
    #f(−3.1302468,−1.5821422) = −106.7645367
    f= np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
        + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
        + (x_0-x_1)**2
    return(f)

#### ackely function ###
def function_2(x_0,x_1):
    f = -20*np.exp(-0.2*np.sqrt(0.5*(x_0**2+x_1**2))) \
        - np.exp(0.5*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))) \
        + np.e + 20
    return(f)

### rastrigin function ###
def function_3(x_0,x_1):
    f = 20 + x_0**2 + x_1**2 - 10*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))
    return(f)

### rosenbrock function ###
def function_4(x_0,x_1):
    f = 100*(x_1-x_0**2)**2 + (1-x_0)**2
    return(f)

### holder table function ###
def function_5(x_0,x_1):
    f = -np.abs(np.sin(x_0)*np.cos(x_1)*np.exp(np.abs(1-np.sqrt(x_0**2+x_1**2)/np.pi)))
    return(f)

#######################################################################################


def main() -> None:
    Test(function_0).run()
    #Test(function_1).run()
    #Test(function_2).run()
    #Test(function_3).run()
    #Test(function_4).run()
    #Test(function_5).run()



if __name__ == "__main__":
    main()



