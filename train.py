
from nn import MLP

# XOR dataset
xs = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]

ys = [0, 1.0, 1.0, 0.0] # XOR targets

mlp = MLP(2, [3, 1])

for k in range(400):
    # forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    
    # reset gradients
    for p in mlp.parameters():
        p.grad = 0.0
        
    # backward pass
    loss.backward()

    # update (gradient descent)
    for p in mlp.parameters():
        p.data += -0.05* p.grad
    
    if k % 10 == 0:
        print(k, loss.data)

for x in xs:
    pred = mlp(x)
    print(f"input: {x} - pred: {pred.data}")