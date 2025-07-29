
import numpy as np
import matplotlib.pyplot as plt
from neuronabinaria import NeuronaBinaria 
# z = w1*x1 + w2*x2 + b
        
# Entradas y etiquetas

Y = np.array([1, 0, 0, 1])

# Parámetros
n1 = NeuronaBinaria(np.array([1, 1, 1, 0]))
n2 = NeuronaBinaria(np.array([0, 1, 1, 1]))


h1 = n1.entrenar()
n1.visualizar()
h2 = n2.entrenar()
x = np.column_stack([h1,h2])
print(x)
n3 = NeuronaBinaria(Y,x)
n3.entrenar()
n3.visualizar()


