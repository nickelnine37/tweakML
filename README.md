# tweakML

TweakML is a python library for building custom machine learning, statistical and general 
mathematical models. Typically, models of this nature can be understood as a large 
function mapping inputs (data, hyperparameters etc.) to outputs (weights, predictions, error 
scores etc.). A common task is to change one or more the inputs to see 
the effect on the output. This often requires recomputing the whole model from scratch or 
complex accounting to keep track of what parts of the model need recomputing and what don't. 

TweakML is designed to silently handle this process by automatically building a 
model dependency graph. That way, when an input is changed, the output can be recomputed in a 
way that is maximally efficient without having to 

## Example: Ridge Regression

Consider the example of ridge regression where we have a feature matrix $\mathbf{X} \in 
\mathbb{R}^{N \times M}$ and an observed target vector $\mathbf{y} \in \mathbb{R}^{N}$. In addition, we 
have a  parameter $\alpha$ which provides regularisation. The coefficient vector $\mathbf
{w} \in \mathbb{R}^{M}$, which we write as a function of $\alpha$, is given by 

$$\mathbf{w}(\alpha) = \left( \mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I}\right)^{-1} \mathbf{X}^\top 
\mathbf{y}$$

The predicted output, $\bar{\mathbf{y}}$, on a validation set $\bar{\mathbf{X}}$ would then by given 

$$\bar{\mathbf{y}} = \bar{\mathbf{X}} \mathbf{w}(\alpha)$$

A python implementation of this might look something like this. 

```python 
import numpy as np

class RidgeRegression:
    
    def __init__(self, X, y, alpha):
        self.X = X
        self.y = y
        self.alpha = alpha
        
    def w(self):
        XTX = self.X.T @ self.X 
        XTy = self.X.T @ self.y
        I = np.eye(self.X.shape[1]) 
        return np.linalg.solve(XTX + self.alpha * I, XTy)
    
    def predict(self, X_):
        return X_ @ self.w()
```

Note that if we change $\alpha$, there is no need to recompute $\mathbf{X}^\top \mathbf{X}$ or $\mathbf{X}^\top 
\mathbf{y}$

Note how the computation of `w` can be visualised as a dependency graph. 

```mermaid
graph TD
X --> XTX
X --> XTy
X --> I
alpha --> alpha * I
I --> alpha * I
y --> XTy
alpha * I --> w
XTX --> w
XTy --> w
```

If `alpha` is changed, there is no need to recompute `XTX` or `XTy` - only the nodes downstream of 
`alpha` need to be recomputed. 

# Building a tweakML Model

TweakML handles this automatically as follows. 

```python
from tweakml import Model, node, Tweakable

class RidgeRegression(Model):
    
    X = Tweakable()
    y = Tweakable()
    alpha = Tweakable()
    
    def __init__(self, X, y, alpha):
        super().__init__()
        self.X = X
        self.y = y
        self.alpha = alpha
    
    @node    
    def XTX(self):
        return self.X.T @ self.X
    
    @node
    def XTy(self):
        return self.X.T @ self.y
    
    @node 
    def I(self):
        return np.eye(self.X.shape[1])
    
    @node
    def alphaI(self):
        return self.alpha * self.I()
	
    @node
    def w(self):
        return np.linalg.solve(self.XTX() + self.alphaI(), self.XTy())
    
    def predict(self, X_):
        return X_ @ self.w()
```

As visible, there are three key steps to making a tweakML model: 

1. Make the model inherit from the `Model` class and call `super().__init__()`. 
2. Define the tweakable parameters at the class level, and set their initial values in the `__init__` method. 
3. Define each step in the computation by writing a method and decorating it with the `node` decorator. 

Now when we run the following code, the intermediate steps in the computation graph are cached. Every time we reset `alpha`, only the nodes downstream are unchached, meaning the model can be recomputed in the most efficient way possible. 

```python 
model = RidgeRegression(X, y, 0.1)

err = []
for alpha in np.linspace(0.01, 1, 50):
    model.alpha = alpha
    err.append(((model.predict(X_) - y_) ** 2).sum()) 
```

