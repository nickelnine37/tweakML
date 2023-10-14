from functools import partial
from collections import defaultdict
from typing import Optional, Any, Type


class Model:
    """
    Base Model class from which all models inherit from. This handles the node accounting logic.
    """

    def __init__(self):
        self.nodes: list[BaseNode] = []        # list of tweakable and derived nodes

        # self.watch signifies which node we are currently watching to register subsequent calls against
        # begins as None and should end as None when the computation graph has complete
        self.watch: Optional[BaseNode] = None

    def add_node(self, node: 'BaseNode'):
        """
        Add a node to the model, either tweakable or a method node
        """
        if node not in self.nodes:
            self.nodes.append(node)

    def register_call(self, node: 'BaseNode'):
        """
        Register that the value at `node` has been accessed/computed while watching `self.watch`. This means
        we should add an edge in the graph to reflect this dependency.
        """
        if self.watch is not None and node not in self.watch.parents:
            self.watch.parents.append(node)
            node.children.append(self.watch)


class CallListener:
    """
    Context manager to listen for subsequent `DerivedNode` calls
    """

    def __init__(self, node: 'DerivedNode'):
        self.node = node
        self.model = node.model

    def __enter__(self):
        self.watch_init = self.model.watch    # which node were we initially watching
        self.model.register_call(self.node)   # register the fact that this derived node has been called
        self.model.watch = self.node          # start watching the current node

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.watch = self.watch_init   # set watch back to the original node


class BaseNode:
    """
    Base class for tweakable and method nodes
    """

    def __init__(self):
        self.model = None
        self.name: Optional[str] = None
        self.cached: bool = False
        self.value: Optional[Any] = None
        self.parents: list['BaseNode'] = []
        self.children: list['BaseNode'] = []

    def uncache(self):
        """
        Uncache this node, and all of its children
        """
        self.cached = False
        self.value = None
        for parent in self.children:
            parent.uncache()

    def __repr__(self):
        return f'Node({self.name}, cached={self.cached})'


class DerivedNode(BaseNode):
    """
    Class representing a node with a value that is derived from other nodes
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.name = func.__name__

    def __get__(self, model: Model, model_cls: Type[Model]):
        """
        Run whenever we access the method. Attach the node to the model if not already done
        """
        if self.model is None and model is not None:
            self.model = model
            self.model.add_node(self)
        return self

    def __call__(self, *args, **kwargs):

        with CallListener(self):

            if not self.cached:
                self.value = self.func(self.model, *args, **kwargs)

        self.cached = True
        return self.value


class Tweakable(BaseNode):
    """
    A tweakable node
    """

    def __init__(self):
        super().__init__()
        self.value = None

    def __set_name__(self, owner, name):
        """
        When we call X = Tweakable(), save 'X' as the node name
        """
        self.name = name

    def __get__(self, model: Model, model_cls: Type[Model]):
        """
        Run whenever we access the value at a node. Attach the node to a model if not already done, and
        register a call to this node
        """
        if self.model is None and model is not None:
            self.model = model
            self.model.add_node(self)
        self.model.register_call(self)
        return self.value

    def __set__(self, model: Model, value: Any):
        """
        When the value at a Tweakable node is set, cache the new value, and uncache the value at all child nodes
        """
        self.value = value
        self.cached = True
        for parent in self.children:
            parent.uncache()


def node(func):
    """
    Simple decorator to turn a `Model` method into a node
    """
    return DerivedNode(func)



