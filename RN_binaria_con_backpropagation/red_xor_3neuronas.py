import numpy as np
import matplotlib.pyplot as plt
from neuronabinaria import NeuronaBinaria 
# z = w1*x1 + w2*x2 + b
        
# Entradas y etiquetas
X = np.array([
[0, 0],
[0, 1],
[1, 0],
[1, 1],
])
Y = np.array([1, 1, 0, 1])

# Parámetros
n1 = NeuronaBinaria(X=X,Y=Y)
n2 = NeuronaBinaria(X=X,Y=Y)
error = True
# Primera ejecución red neuronal y creacion de la neurona de salida
n1.calc_prediccion()
n2.calc_prediccion()
input = [n1,n2]
n3 = NeuronaBinaria(Y,input)
n3.calc_prediccion()
# Miramos si coincide la prediccion de n3 con Y
if all(np.equal((n3.array_predicciones>=0.5).astype(int),Y)):
    error = False
while(error):
    n3.corregir()
    n1.corregir(ultima=n3)
    n2.corregir(ultima=n3)
    n1.calc_prediccion()
    n2.calc_prediccion() 
    n3.calc_prediccion()
    if all(np.equal((n3.array_predicciones>=0.5).astype(int),Y)):
        error = False
        
    
    
        
        
n3.visualizar()

