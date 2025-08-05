import numpy as np
import matplotlib.pyplot as plt
from redneuronal import RedNeuronal
# z = w1*x1 + ... + wn*xn + b
        
# Entradas y etiquetas
# El patrón es que el primer elemento del array es el resultado, falta uno de los casos el cual luego se intentara predecir
# con lo que aprende del resto
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1], 
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1]  
])
Y = np.array([0,0,0,0,1,1,1])
pruebas = []

#Aqui creo 100 redes neuronales con solo una capa de 20 neuronas y 1 salida, y les pongo un nuevo caso para las cuales no estan entrenadas.
#Se puede ver que una media de 95 son capaces de predecir que el resultado es "1", para el caso que nunca vieron
for i in range(100):
    rn = RedNeuronal(X,Y,[20,1])
    rn.entrenar()
    pruebas.append(rn.predecir([1, 1, 0])[0])
    
print("Predijo el resultado: ",sum(pruebas))


        
    
    
        
        


