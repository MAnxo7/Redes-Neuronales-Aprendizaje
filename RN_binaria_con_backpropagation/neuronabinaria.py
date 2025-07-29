import numpy as np
import matplotlib.pyplot as plt
import math
class NeuronaBinaria:

    w1 = 1
    w2 = 1
    b = 0.0
    lr = 0.1
    it = 0
    error = True
    array_predicciones = None
    input = None
    deltaw1 = None
    deltaw2 = None
    X = None
    
    
    def __init__(self, Y, input=None, X=None):
        self.Y = Y
        if X is None:
            self.input = input
            self.X = np.column_stack([self.input[0].array_predicciones,self.input[1].array_predicciones])           
        if input is None:
            self.X = X

    
    def calc_prediccion(self):    
        self.array_predicciones = self.w1*self.X[:,0] + self.w2*self.X[:,1] + self.b
        self.__f_act(self.array_predicciones)
        return self.array_predicciones
    
    def corregir(self,ultima=False):
        if not ultima:
            self.deltaw1 = (self.array_predicciones - self.Y) * self.__der_f_act(self.X[:,0])
            self.deltaw2 = (self.array_predicciones - self.Y) * self.__der_f_act(self.X[:,1])
        else:
            self.deltaw1 = ultima.w1 * ultima.deltaw1 *  self.__der_f_act(self.X[:,0])
            self.deltaw2 = ultima.w2 * ultima.deltaw2 *  self.__der_f_act(self.X[:,1])
        self.w1 -= sum(self.lr * self.deltaw1 * self.X[:,0])
        self.w2 -= sum(self.lr * self.deltaw2 * self.X[:,1])
        self.b  -= self.lr * (self.deltaw1 + self.deltaw2)
            
            
        
    def visualizar(self):
        print(f"Error: {self.error} Iteraciones: {self.it}")        
        print((self.array_predicciones>=0.5).astype(int))    
        print(f"w1: {self.w1} w2: {self.w2} b: {self.b}")

        
    def __f_act(self,x):
        for i in range (0,x.size) :
           x[i] = 1 / (1 + math.exp(-x[i]))
           
    def __der_f_act(self,x):
        for i in range (0,x.size) :
           x[i] = math.exp(-x[i]) / ((1 + math.exp(-x[i]))**2)
        return x

                
    