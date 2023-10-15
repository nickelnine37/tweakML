from tweakml.models import RidgeRegression
from tweakml.tweakml import DerivedNode, Tweakable, BaseNode, Model
import numpy as np


def get_test_model() -> (RidgeRegression, np.ndarray):

    np.random.seed(9)

    N = 5
    M = 3

    X = np.random.randn(N, M)
    y = np.random.randn(N)

    alpha = 1
    mod = RidgeRegression(X, y, alpha)

    return mod, X


def test_caching():

    mod, X = get_test_model()

    mod.predict(X)

    print(mod.nodes)

    # after prediction, all nodes should be cached
    for node in mod.nodes:
        assert node.cached

    mod.alpha = 2

    # after setting alpha, only nodes downstream of alpha should be uncached
    for node in mod.nodes:
        if is_downstream(node, mod.get_node('alpha')):
            assert not node.cached
        else:
            assert node.cached


def is_downstream(node1: BaseNode, node2: BaseNode):
    """
    Check whether node1 is downstream of node2
    """

    if len(node1.parents) == 0:
        return False

    if node2 in node1.parents:
        return True

    return any(is_downstream(parent, node2) for parent in node1.parents)


# def test_repr():
#
#     mod, X = get_test_model()
#
#     mod.predict(X)
#
#     print(mod.nodes[0])