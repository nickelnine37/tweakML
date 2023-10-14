from tweakml.graph import node, Tweakable, Model
import numpy as np


class RidgeRegression(Model):

    # X = Tweakable()
    # y = Tweakable()
    alpha = Tweakable()

    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float):
        super().__init__()
        self.X = X
        self.y = y
        self.alpha = alpha

    @node
    def _decompose(self):
        return np.linalg.eigh(self.X.T @ self.X)

    @node
    def lam(self):
        return self._decompose()[0]

    @node
    def U(self):
        return self._decompose()[1]

    @node
    def lamAlInv(self):
        return (self.lam() + self.alpha) ** -1

    @node
    def M(self):
        return (self.U() * self.lamAlInv()) @ self.U().T

    @node
    def Xy(self):
        return self.X.T @ self.y

    @node
    def w(self):
        return self.M() @ self.Xy()

    def predict(self, X_: np.ndarray):
        return X_ @ self.w()


if __name__ == '__main__':

    from pprint import pprint

    np.random.seed(9)

    N = 5
    M = 3

    X = np.random.randn(N, M)
    y = np.random.randn(N)

    alpha = 1
    mod = RidgeRegression(X, y, alpha)

    mod.predict(X)

    for nd in mod.nodes:
        print(nd)
        print(f'\tChildren: {[child for child in nd.children]}')
        print(f'\tParents: {[parent for parent in nd.parents]}')

    mod.alpha = 2
    print('\nSetting alpha\n')

    for nd in mod.nodes:
        print(nd)
        print(f'\tChildren: {[child for child in nd.children]}')
        print(f'\tParents: {[parent for parent in nd.parents]}')

    mod.predict(X)

    #
    # pprint(mod.graph.nodes)
    # pprint(mod.graph.edges)
    #
    # mod.w.tweak(5)
    #
    # print(mod.w())
