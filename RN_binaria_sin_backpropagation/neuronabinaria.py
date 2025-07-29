import numpy as np
import matplotlib.pyplot as plt
class NeuronaBinaria:
    X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    ])
    w1 = 1
    w2 = 1
    b = 0.0
    lr = 0.1
    it = 0
    error = True
    array_predicciones = None
    
    def __init__(self, Y, X=None):
        self.Y = Y
        if X is not None:
            self.X = X

    
    def entrenar(self):
 
        while(self.error and self.it < 200):
            #Intentamos predecir un resultado
            self.array_predicciones = self.w1*self.X[:,0] + self.w2*self.X[:,1] + self.b
            self.__f_act(self.array_predicciones)
            #Aprendizaje, el array delta es una array de error el cual castiga x1 y x2 correspondiente 
            array_delta = self.Y - self.array_predicciones
            self.w1+=self.lr*sum(array_delta * self.X[:,0])
            self.w2+=self.lr*sum(array_delta * self.X[:,1])
            self.b+=self.lr*sum(array_delta)
            #Comprobacion de error        
            if np.all(self.array_predicciones == self.Y):
                self.error = False
            self.it+=1
        return self.array_predicciones
    
    def visualizar(self):
        print(f"Error: {self.error} Iteraciones: {self.it}")        
        print(self.array_predicciones)    
        print(f"w1: {self.w1} w2: {self.w2} b: {self.b}")

        plt.scatter(
        NeuronaBinaria.X[self.array_predicciones == 0, 0], NeuronaBinaria.X[self.array_predicciones == 0, 1], marker='o', label='Clase 0'
        )
        plt.scatter(
            NeuronaBinaria.X[self.array_predicciones == 1, 0], NeuronaBinaria.X[self.array_predicciones== 1, 1],
            marker='x', label='Clase 1'
        )

        x_vals = np.array([-0.2, 1.2])
        y_vals = -(self.w1 * x_vals + self.b) / self.w2
        plt.plot(x_vals, y_vals, linestyle='--', label='Frontera')
            
            
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Visualización de clasificación binaria')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def __f_act(self,x):
        for i in range (0,x.size) :
            if x[i] > 0 :
                x[i] = 1
            else :
                x[i] = 0
                
    