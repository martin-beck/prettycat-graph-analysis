import abc
import itertools
import uuid
import typing

import lxml.etree


class AbstractNode:
    def __init__(self, unique_id):
        super().__init__()
        self._unique_id = unique_id

    @property
    def unique_id(self) -> str:
        """
        The project-global unique identifier of the node.
        """
        return self._unique_id


class Node(AbstractNode):
    def __init__(self, unique_id):
        super().__init__(unique_id)
        self._inputs = []
        self._predecessors = []
        self._successors = []
        self._block = None

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def inputs(self) -> typing.Iterable[str]:
        """
        A list of unique IDs of input nodes.
        """
        return iter(self._inputs)

    @property
    def successors(self) -> typing.Iterable['Node']:
        return iter(self._successors)

    @property
    def predecessors(self) -> typing.Iterable['Node']:
        return iter(self._predecessors)

    @property
    def block(self) -> 'BasicBlock':
        return self._block


class CallNode(Node):
    @property
    def call_target(self) -> str:
        """
        Unique ID of the call target.
        """

    @property
    def parameters(self) -> typing.Iterable[typing.Tuple[str, str]]:
        """
        List of parameters to pass.
        """


class BasicBlock(AbstractNode):
    def __init__(self, unique_id):
        super().__init__(unique_id)
        self._nodes = []

    @property
    def nodes(self) -> typing.Iterable[Node]:
        """
        List of nodes in this basic block.
        """
        return iter(self._nodes)

    @property
    def exits(self) -> typing.Iterable['BasicBlock']:
        """
        List of basic block nodes to which this basic block may jump.
        """
        if not self._nodes:
            return iter([])
        return (successor.block for successor in self._nodes[-1].successors)


class ControlDataFlowGraph:
    def __init__(self):
        super().__init__()
        self._blocks = []
        self._floating_nodes = []

    @property
    def blocks(self) -> typing.Iterable[BasicBlock]:
        return iter(self._blocks)

    @property
    def nodes(self) -> typing.Iterable[Node]:
        return itertools.chain(
            self._floating_nodes,
            *(block.nodes for block in self._blocks)
        )

    def block_by_node(self, node: Node) -> BasicBlock:
        """
        Return the block to which a node belongs.

        :raises KeyError: if the node does not belong to any basic block or
            does not belong to this :class:`ControlDataFlowGraph`.
        """
        for block in self._blocks:
            if node in block.nodes:
                return block
        raise KeyError(node)

    def node_by_id(self, id_: str) -> Node:
        """
        Return a node by its unique ID.

        :param id_: The ID of the node to return.
        :type id_: :class:`str`
        :raises KeyError: if no node with `id_` exists in the graph.
        :rtype: Node
        :return: The node identified by the ID.
        """
        for node in self.nodes:
            if node.unique_id == id_:
                return node
        raise KeyError(id_)

    def new_block(self, *, id_: typing.Optional[str] = None) -> BasicBlock:
        """
        Create a new basic block.
        """
        if not id_:
            id_ = "urn:uuid:" + str(uuid.uuid4())

        bb = BasicBlock(id_)
        self._blocks.append(bb)
        return bb

    def new_node(self, class_, *,
                 block: typing.Optional[BasicBlock] = None,
                 id_: typing.Optional[str] = None) -> Node:
        """
        Create a new node.
        """
        if not id_:
            id_ = "urn:uuid:" + str(uuid.uuid4())

        node = class_(id_)
        node._block = block
        if block:
            if block._nodes:
                node._predecessors.append(block._nodes[-1])
                block._nodes[-1]._successors.append(node)
            block._nodes.append(node)
        else:
            self._floating_nodes.append(node)
        return node

    def add_successor(self, at: Node, successor: Node):
        """
        Add a control flow successor to a node.

        :param at: The node to which to add a successor.
        :type at: :class:`Node`
        :param successor: The node to add as successor.
        :type successor: :class:`Node`
        :raises ValueError: if `at` is not at the end of a block.
        :raises ValueError: if `successor` is not at the beginning of a block.

        .. note::

            To split a basic block, use :meth:`split_block`.
        """
        try:
            block_at = self.block_by_node(at)
        except KeyError as exc:
            raise ValueError(
                "cannot add successor to floating node"
            ) from exc
        try:
            block_successor = self.block_by_node(successor)
        except KeyError as exc:
            raise ValueError(
                "cannot add floating node as successor"
            ) from exc

        if block_at._nodes[-1] is not at:
            raise ValueError(
                "cannot add successor to non-last node"
            )

        if block_successor._nodes[0] is not successor:
            raise ValueError(
                "cannot add non-first node as successor"
            )

        at._successors.append(successor)
        successor._predecessors.append(at)

    def split_block(self, at_node: Node):
        """
        Split the basic block of a node such that the node and its successors
        are in a new basic block.

        :param at_node: The first node to be contained in a new basic block.
        :type at_node: :class:`Node`
        :raises ValueError: if `at_node` does not belong to any basic block
        :raises ValueError: if `at_node` is the first node in a basic block
        :return: The newly created block.
        :rtype: :class:`BasicBlock`
        """
        block = self.new_block()
        old_block = at_node.block
        old_index = old_block._nodes.index(at_node)
        block._nodes = old_block._nodes[old_index:]
        del old_block._nodes[old_index:]

        for node in block._nodes:
            node._block = block

        return block

    def inline_call(self, call_node: Node, method_cdfg):
        """
        Inline a given call node.

        :param call_node: The call to inline.
        :type call_node: :class:`CallNode`
        :param method_cdfg: The control/data flow graph of the called method.
        :type method_cdfg: :class:`ControlDataFlowGraph`

        The control/data flow graph of the inlined method will be merged into
        this contorl/data flow graph. The unique IDs are relabeled to allow
        to inline the same method at different callsites.

        The relabeling uses a new UUID for each inlining. It is the callers
        responsibility to avoid recursive inlining.
        """

    def load_from_xml(self, tree: lxml.etree.Element):
        """
        Load the control/data flow graph from an XML element.
        """
