import numpy as np

# y = w*x + b

#Declaración de variables 
array_peso = np.array([1,2,3,4]) #x
array_precio = np.array([3,6,9,12]) #y

w = 1
b = 1

lr = 0.1
error = 99999999999
error_min = 1e-6

max_it = 10000
it = 0

while error > error_min and it < max_it:
    #Predicción, obtengo el error cuadratico medio de lo predicho respecto a lo esperado
    array_y = w*array_peso + b
    array_errorcuadratico = pow(( array_precio - array_y ),2) 

    error = np.mean(array_errorcuadratico) 

    #Mediante la derivada se obtiene a donde tiene que dirigirse el siguiente w y b
    array_w2 = array_peso * (array_precio - array_y)
    array_b2 = (array_precio - array_y)

    grad_w2 = -2 * np.mean(array_w2)
    grad_b2 = -2 * np.mean(array_b2)

    w = w - lr * grad_w2
    b = b - lr * grad_b2
    
    it+=1

print(f"Convergio en {it} iteraciones")
print(f"w = {w:.4f}, b = {b:.4f}, error final = {error:.6f}")