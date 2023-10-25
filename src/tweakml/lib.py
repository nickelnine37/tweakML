from typing import Optional, Any, Type, Callable
from types import MethodType


class Model:
    """
    Base Model class from which all models inherit from. This handles the node accounting logic.
    """

    # holds a list of nodes for the model. These are added in __set_name__ at each node
    nodes: list['BaseNode'] = []

    def __init__(self):

        # self.watch signifies which node we are currently watching to register subsequent calls against
        # begins as None and should end as None when the computation graph has complete
        self.watch: Optional[BaseNode] = None

        # this holds the cached values at each node
        self.values: dict[str, Any] = {}

        # this tells us whether the node is cached or not
        self.cached: dict[str, bool] = {}

        # initialise values and cache statuses
        for node in self.nodes:
            self.uncache(node)
            if isinstance(node, Tweakable):

                def closure(node: 'Tweakable'):
                    def tweak(value):
                        node.__set__(self, value)
                        return self

                    return tweak

                self.__dict__[f'set_{node.name}'] = closure(node)

    def cache(self, node: 'BaseNode', value: Any):
        self.values[node.name] = value
        self.cached[node.name] = True

    def uncache(self, node: 'BaseNode'):
        self.values[node.name] = None
        self.cached[node.name] = False

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

    def __init__(self, node: 'DerivedNode') -> None:
        self.node = node
        self.model_inst = node.model_inst

    def __enter__(self) -> None:
        self.watch_init = self.model_inst.watch    # which node were we initially watching
        self.model_inst.register_call(self.node)   # register the fact that this derived node has been called
        self.model_inst.watch = self.node          # start watching the current node

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.model_inst.watch = self.watch_init   # set watch back to the original node


class BaseNode:
    """
    Base class for tweakable and method nodes
    """

    def __init__(self):
        self.name: Optional[str] = None
        self.parents: list[BaseNode] = []
        self.children: list[BaseNode] = []

    def uncache(self, model_inst: Model) -> None:
        """
        Uncache this node with respect to a model instance, and all of its children
        """
        model_inst.uncache(self)
        for child in self.children:
            child.uncache(model_inst)

    def __set_name__(self, model_cls: Type[Model], name: str) -> None:
        """
        When we call X = Tweakable(), or define use the @node decorator, name the resultant node
        by the variable or decorated method name. Also add the node to the nodes list for the class.
        This should run only once when the node class is defined.
        """
        self.name = name
        model_cls.nodes.append(self)

    def __repr__(self) -> str:
        return f'Node({self.name})'


class DerivedNode(BaseNode):
    """
    Class representing a node with a value that is derived from other nodes
    """

    def __init__(self, func) -> None:
        super().__init__()
        self.func: Callable = func
        self.model_inst: Optional[Model] = None

    def __get__(self, model_inst: Model, model_cls: Type[Model]) -> 'DerivedNode':
        """
        Run whenever we access the method. Attach the node to the model if not already done
        """
        if model_inst is None:
            raise ValueError('')
        self.model_inst = model_inst
        return self

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the function associated with this derived node. Listen for subsequent node calls. Return
        the cached value if available.
        """

        with CallListener(self):

            if not self.model_inst.cached[self.name]:
                value = self.func(self.model_inst, *args, **kwargs)
                self.model_inst.cache(self, value)
            else:
                value = self.model_inst.values[self.name]

        self.model_inst = None
        return value


class Tweakable(BaseNode):
    """
    A tweakable node
    """

    def __init__(self):
        super().__init__()

    def __get__(self, model_inst: Model, model_cls: Type[Model]) -> Any:
        """
        Run whenever we access the value at a node. Attach the node to the model instance, register a call
        and return the cached value
        """
        model_inst.register_call(self)
        if model_inst.cached[self.name]:
            return model_inst.values[self.name]
        else:
            raise AttributeError(f'The value at node {self.name} has not been set')

    def __set__(self, model_inst: Model, value: Any) -> None:
        """
        When the value at a Tweakable node is set, cache the new value, and uncache the value at all child nodes
        as they will need to be recomputed
        """
        model_inst.cache(self, value)
        for child in self.children:
            child.uncache(model_inst)


def node(func) -> DerivedNode:
    """
    Simple decorator to turn a `Model` method into a node
    """
    return DerivedNode(func)



