import numpy as np
import matplotlib.pyplot as plt
import math
class NeuronaBinaria:
    w = None #array_de_pesos
    siguientes = None #Neuronas posteriores a esta
    X = None #matriz de datos entrante
    y = None #resultado esperado
    z = None #predicción
    z_act = None #prediccion_activada
    z_act_der = None 
    lr = 0.1 #learning_rate
    b = 0 #sesgo
    delta = None # Variable que estima cuanto se debe corregir
    
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.w = None 
        self.siguientes = None
        self.z = None 
        self.z_act = None 
        self.z_act_der = None 
        self.lr = 0.1 
        self.b = 0 
        self.delta = None 
    
    def calcular(self):
        self.z = self.X@self.w + self.b
        self.z_act = self.__f_act(self.z)
        self.z_act_der = self.__der_f_act(self.z)                  
        
    def corregir(self,n):
        if not self.siguientes:
            self.delta = (self.z_act-self.y)*self.z_act_der
        else:
            sumt = 0
            for i in range(0,len(self.siguientes)):
                sumt+=self.siguientes[i].w[n]*self.siguientes[i].delta
            self.delta = self.z_act_der * sumt
        
        for i in range(0,self.w.size):
            self.w[i]-=sum(self.lr*self.X[:,i]*self.delta)
        self.b-=self.lr*np.mean(self.delta)
                  


        
    def __f_act(self,z):
        z = np.array(z, dtype=np.float64)
        return np.tanh(z)
        
           
    def __der_f_act(self,z):
        return 1.0 - np.tanh(z) ** 2

                
    