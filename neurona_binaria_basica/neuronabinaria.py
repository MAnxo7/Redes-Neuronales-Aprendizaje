import numpy as np

# z = w1*x1 + w2*x2 + b
 
def f_act(x):
    for i in range (0,x.size) :
        if x[i] > 0 :
            x[i] = 1
        else :
            x[i] = 0
        

# Entradas y etiquetas
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
Y = np.array([0, 1, 1, 1])

# Parámetros
w1 = 1
w2 = 1
b = 0.0
lr = 0.1
max_its = 20
it = 0
error = True

while(error and it < 200):
    #Intentamos predecir un resultado
    array_predicciones = w1*X[:,0] + w2*X[:,1] + b
    f_act(array_predicciones)
    #Aprendizaje
    array_delta = Y - array_predicciones
    w1+=lr*sum(array_delta * X[:,0])
    w2+=lr*sum(array_delta * X[:,1])
    b+=lr*sum(array_delta)
    #Comprobacion de error
    if np.all(array_predicciones == Y):
        error = False;
    it+=1
    
print(f"Error: {error} Iteraciones: {it}")        
print(array_predicciones)    
print(f"w1: {w1} w2: {w2} b: {b}")