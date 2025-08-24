import numpy as np
import matplotlib.pyplot as plt
import math
from neuronabinaria import NeuronaBinaria 

class RedNeuronal:
    n_entradas = None #Columnas de X
    n_neuronas_capas = None # Numero de neuronas por capa, un array numerico
    neuronas_capas = None # Array de objetos neurona que es la red neuronal en si
    output = None #Salida de la red neuronal, un array, normalmente de 1 elemento
    X = None
    y = None
    
    #Para inicializar una red neuronal necesitamos:
    # X: matriz con datos
    # y: array con la salida esperada para cada dato
    # n_neuronas_capas: un array con el numero de neuronas por capa, siempre el ultimo elemento tiene que ser "1", 
    # debido a que la arquitectura no esta preparada para recibir más de 1 salida
    def __init__(self,X,y,n_neuronas_capas):
        self.X = X
        self.y = y
        self.n_entradas = len(np.column_stack(X))
        self.n_neuronas_capas = n_neuronas_capas
        self.neuronas_capas = []
        
        for i in range(0,len(self.n_neuronas_capas)):
            self.neuronas_capas.append([]) 
            
        for i in range(len(self.n_neuronas_capas)-1, -1, -1): #Recorre los indices al reves y va creando neuronas en cada capa           
            capa = [NeuronaBinaria(self.X, self.y) for _ in range(n_neuronas_capas[i])]
            self.neuronas_capas[i] = capa
            #Creamos los pesos para cada neurona
            for j in range(0,self.n_neuronas_capas[i]):
                if i == 0:
                    self.neuronas_capas[i][j].w = np.random.uniform(-1.0, 1.0, size=(self.n_entradas,))
                else:
                    self.neuronas_capas[i][j].w = np.random.uniform(-1.0, 1.0, size=(self.n_neuronas_capas[i-1],)) 
                    self.neuronas_capas[i][j].X = np.full((len(self.y),self.n_neuronas_capas[i-1]), None, dtype=np.float64)
            #Si la neurona no es la ultima, se asigna una lista de siguientes a cada neurona recien creada
            if i < len(self.n_neuronas_capas)-1:
                for neurona in self.neuronas_capas[i]:
                    neurona.siguientes = self.neuronas_capas[i+1]
                    
        self.output = self.neuronas_capas[len(self.neuronas_capas)-1][0]
                    



                
    def entrenar(self):
        error = True
        its = 0
        #QUE EL INDICE DE X DE LA SIGUIENTE CORRESPONDA AL INDICE ACTUAL
        while error and its < 2500:
            for i in range(0,len(self.neuronas_capas)):
                for j in range(0,len(self.neuronas_capas[i])):
                    self.neuronas_capas[i][j].calcular()
                    #Si la neurona tiene siguientes, pone en el X de las siguientes su correspondiente Z
                    if self.neuronas_capas[i][j].siguientes:
                        for neurona in self.neuronas_capas[i][j].siguientes:

                            neurona.X[:,j] = self.neuronas_capas[i][j].z_act 

            error = self.__binario(self.output.z_act) != self.y
            error=any(error)
            if error:
                for i in range(len(self.n_neuronas_capas)-1, -1, -1):
                    for j in range(0,self.n_neuronas_capas[i]):
                        self.neuronas_capas[i][j].corregir(j)
            its+=1
        if error:                
            print("FALLO en ",its," iteraciones")
        else:
            print("EXITO en ",its," iteraciones")
        
    def predecir(self,x):
        x = np.array(x)
        for capa in self.neuronas_capas:
            resultado = []
            for neurona in capa:
                neurona.X = x.copy()
                neurona.calcular()
                resultado.append(neurona.z_act)
            x = np.array(resultado)
        return self.__binario(np.array([self.output.z_act]))
            
            
            
        
  
    
    def __binario(self,array):
        toret = array.copy()
        for i in range(0,array.size):
            if array[i]>=0.5:
                toret[i]=1
            else:
                toret[i]=0
        return toret
        

        
                        
                    
        
        
            
                
            

    
     
