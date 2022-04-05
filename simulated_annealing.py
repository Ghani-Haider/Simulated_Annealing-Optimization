import numpy as np
import math
import matplotlib.pyplot as plt

# functions for which to find min/max
def SphereFunction(x,y):
    return (x**2 + y**2)

def RosenbrockFunction(x,y):
    return ((100*(x**2 - y**2)) + (1 - x**2))

def GriewankFunction(x,y):
    return (((x**2 + y**2)/4000) - (math.cos(x) * math.cos(y/math.sqrt(2))) + 1)

# for plotting the co-ordinates
x_lst = []
y_lst = []
output_lst = []

# Simulated Annealing Algo
def SimulatedAnnealing(function, rangeX, rangeY, step=1, Max=True):
    currentX = np.random.uniform(rangeX[0], rangeX[1])
    currentY = np.random.uniform(rangeY[0], rangeY[1])

    temperature = 1000
    temp_min = 0.0000001
    N_iter=100
    factor=0.8

    while temperature > temp_min:
        for j in range(N_iter):
            trialX = np.random.uniform(max(currentX - step, rangeX[0]), min(currentX + step, rangeX[1]))
            trialY = np.random.uniform(max(currentY - step, rangeY[0]), min(currentY + step, rangeY[1]))
            delta = function(trialX, trialY) - function(currentX, currentY)
            
            x_lst.append(currentX)
            y_lst.append(currentY)
            
            # to find maximum
            if Max:
                if delta > 0:
                    currentX = trialX
                    currentY = trialY
                else:
                    m = math.exp(delta/temperature)
                    p = np.random.uniform(0,1)
                    if p < m:
                        currentX = trialX
                        currentY = trialY
            # to find minimum
            else:
                if delta < 0:
                    currentX = trialX
                    currentY = trialY
                else:
                    m = math.exp(-delta/temperature)
                    p = np.random.uniform(0,1)
                    if p < m:
                        currentX = trialX
                        currentY = trialY
        temperature = factor * temperature
    
    # for plotting the function's outputs
    for i in range(len(x_lst)):
        output_lst.append(function(x_lst[i], y_lst[i]))
    
    # returning the result
    result = function(currentX, currentY)
    return ([currentX, currentY], result)



def main():
    # simulated annealing function call
    x_range = [-5, 5]
    y_range = [-5, 5]
    pos, res = SimulatedAnnealing(GriewankFunction, x_range, y_range, 1, True) # (function, x_range, y_range, step, True=Max/False=Min)
    print("x = "+str(pos[0])+" y = "+str(pos[1])+" f = "+str(res))

    # graph plotting
    plt.subplot(2,1,1)
    plt.plot(output_lst, c="red", label="f")
    plt.ylabel("Objective")
    plt.legend(loc="lower right")

    plt.subplot(2,1,2)
    plt.plot(x_lst, c="blue", label="x")
    plt.plot(y_lst, c="green", label="y")
    plt.ylabel("Co-ordinates")
    plt.legend(loc="lower right")

    plt.show()

if __name__ == "__main__":
    main()
