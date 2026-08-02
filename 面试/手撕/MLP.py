import numpy as np

x=np.random.randn(5,2)

def relu(x):
    return np.maximum(0, x)

W1=np.random.randn(2,3)
b1=np.zeros(3)

W2=np.random.randn(3,1)
b2=np.zeros(1)

def mlp(x):
    h=x@W1+b1
    h=relu(h)
    y=h@W2+b2
    return y
out=mlp(x)
print(out)
