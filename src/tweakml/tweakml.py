from functools import partial
from collections import defaultdict
from typing import Optional, Any, Type


class Model:
    """
    Base Model class from which all models inherit from. This handles the node accounting logic.
    """

    node_names = []  # list of node names for reference set at class level. Populated on __set__name__

    def __init__(self):
        self.nodes: list[BaseNode] = []        # list of tweakable and derived nodes

        # self.watch signifies which node we are currently watching to register subsequent calls against
        # begins as None and should end as None when the computation graph has complete
        self.watch: Optional[BaseNode] = None

    def add_node(self, node: 'BaseNode') -> None:
        """
        Add a node to the model, either tweakable or a method node
        """
        if node not in self.nodes:
            self.nodes.append(node)

    def get_node(self, name: str) -> 'BaseNode':
        """
        Get a node by name. Here, we want to return the actual node object, not its value
        """
        return self.__class__.__dict__[name]

    def register_call(self, node: 'BaseNode') -> None:
        """
        Register that the value at `node` has been accessed/computed while watching `self.watch`. This means
        we should add an edge in the graph to reflect this dependency.
        """
        if self.watch is not None and node not in self.watch.parents:
            self.watch.parents.append(node)
            node.children.append(self.watch)


class CallListener:
    """
    Context manager to listen for subsequent node calls
    """

    def __init__(self, node: 'DerivedNode'):
        self.node = node
        self.model_inst = node.model_inst

    def __enter__(self):
        self.watch_init = self.model_inst.watch    # which node were we initially watching
        self.model_inst.register_call(self.node)   # register the fact that this derived node has been called
        self.model_inst.watch = self.node          # start watching the current node

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model_inst.watch = self.watch_init   # set watch back to the original node


class BaseNode:
    """
    Base class for tweakable and method nodes
    """

    def __init__(self):
        self.model_inst = None
        self.name: Optional[str] = None
        self.cached: bool = False
        self.value: Optional[Any] = None
        self.parents: list[BaseNode] = []
        self.children: list[BaseNode] = []

    def uncache(self):
        """
        Uncache this node, and all of its children
        """
        self.cached = False
        self.value = None
        for child in self.children:
            child.uncache()

    def attach(self, model_inst: Model):
        """
        Attach this node to a model instance
        """
        if self.model_inst is None and model_inst is not None:
            self.model_inst = model_inst
            self.model_inst.nodes.append(self)

    def __set_name__(self, model_cls: Type[Model], name: str) -> None:
        """
        When we call X = Tweakable(), save 'X' as the node name
        """
        self.name = name
        model_cls.node_names.append(name)

    def __repr__(self):
        return f'Node({self.name}, cached={self.cached})'


class DerivedNode(BaseNode):
    """
    Class representing a node with a value that is derived from other nodes
    """

    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def __get__(self, model_inst: Model, model_cls: Type[Model]) -> Any:
        """
        Run whenever we access the method. Attach the node to the model if not already done
        """
        self.attach(model_inst)
        return self

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the function associated with this derived node. Listen for subsequent node calls. Return
        the cached value if available.
        """

        with CallListener(self):

            if not self.cached:
                self.value = self.func(self.model_inst, *args, **kwargs)

        self.cached = True
        return self.value


class Tweakable(BaseNode):
    """
    A tweakable node
    """

    def __init__(self):
        super().__init__()
        self.value = None

    def __get__(self, model_inst: Model, model_cls: Type[Model]) -> Any:
        """
        Run whenever we access the value at a node. Attach the node to the model instance, register a call
        and return the cached value
        """
        self.attach(model_inst)
        self.model_inst.register_call(self)
        if self.cached:
            return self.value
        else:
            raise AttributeError(f'The value at node {self.name} has not been set')

    def __set__(self, model: Model, value: Any) -> None:
        """
        When the value at a Tweakable node is set, cache the new value, and uncache the value at all child nodes
        as they will need to be recomputed
        """
        self.value = value
        self.cached = True
        for child in self.children:
            child.uncache()


def node(func) -> DerivedNode:
    """
    Simple decorator to turn a `Model` method into a node
    """
    return DerivedNode(func)



