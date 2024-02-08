#import torch
import numpy as np

#f = w * x

#f = 2 * x

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([4,8,12,16], dtype=np.float32)

w = 0.0

#model preciction
def forward(x):
    return w * x

#loss
def loss(y, y_predicted):
    return((y-y_predicted)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N * 2x * (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prefiction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    #prefiction = forward pass
    y_pred = forward(X)
    
    #loss
    l = loss(Y, y_pred)

    #gradient
    dw = gradient(X, Y, y_pred)

    #update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
    
print(f'Prefiction after training: f(5) = {forward(10):.3f}')