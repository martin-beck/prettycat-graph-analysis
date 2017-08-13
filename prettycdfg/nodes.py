import abc
import itertools
import uuid
import typing

import lxml.etree

from .xmlutil import ASM


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
    def inputs(self) -> typing.Iterable['Node']:
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


class ASMNode(Node):
    def __init__(self, unique_id,
                 line: typing.Optional[int] = None,
                 opcode: typing.Optional[int] = None):
        super().__init__(unique_id)
        self._line = line
        self._opcode = opcode

    @property
    def line(self):
        return self._line

    @property
    def opcode(self):
        return self._opcode

    @property
    def local_id(self):
        return self.unique_id.split("/", 1)[1]


class ParameterNode(Node):
    def __init__(self, unique_id,
                 type_: typing.Optional[str] = None):
        super().__init__(unique_id)
        self._type = type_

    @property
    def type_(self):
        return self._type


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


def node_from_xml(tree):
    if tree.tag == IrGraph.node:
        return Node(tree.get("id"))
    raise ValueError("unsupported node: {}".format(tree.tag))


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

    @property
    def floating_nodes(self) -> typing.Iterable[Node]:
        return iter(self._floating_nodes)

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
                 id_: typing.Optional[str] = None,
                 **kwargs) -> Node:
        """
        Create a new node.
        """
        if not id_:
            id_ = "urn:uuid:" + str(uuid.uuid4())

        node = class_(id_, **kwargs)
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

    def add_input(self, at: Node, source: Node):
        """
        Add an input to a node.

        .. note::

            This does not check that the data flow graph is still valid.
        """
        at._inputs.append(source)

    def split_block(self, at_node: Node) -> BasicBlock:
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

    def join_blocks(self, b1: BasicBlock, b2: BasicBlock) -> BasicBlock:
        """
        Join two basic blocks by appending the second to the first.

        :param b1: First basic block
        :type b1: :class:`BasicBlock`
        :param b2: Second basic block
        :type b2: :class:`BasicBlock`
        :raises ValueError: if `b1` and `b2` are the same basic block
        :raises ValueError: if `b1` and `b2` cannot be joined due to control
            flow
        :return: The first basic block
        :rtype: :class:`BasicBlock`

        If `b1` is equal to `b2`, :class:`ValueError` is raised.

        If `b1` or `b2` is empty, the nodes are simply moved to `b1` and `b2`
        is discarded.

        If the last node of `b1` has not exactly one successor or if the
        successor is not the first node of `b2`, :class:`ValueError` is raised.
        If the first node of `b2` has not exactly one predecessor or if the
        predecessor is not the last node of `b1`, :class:`ValueError` is
        raised.
        """
        if b2 not in b1.exits and b2._nodes and b1._nodes:
            raise ValueError("b2 is not a successor of b1")
        if b1._nodes and len(b1._nodes[-1]._successors) > 1:
            raise ValueError("b1 has multiple successors")
        if b2._nodes and len(b2._nodes[0]._predecessors) > 1:
            raise ValueError("b2 has multiple predecessors")
        b1._nodes.extend(b2._nodes)
        b2._nodes.clear()
        self._blocks.remove(b2)
        return b1

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

    def simplify_basic_blocks(self):
        """
        Simplify the basic blocks in the CDFG.

        The basic blocks are expanded as far as possible.
        """

        while True:
            for bb in self._blocks:
                if not bb._nodes:
                    continue
                last = bb._nodes[-1]
                if len(last._successors) != 1:
                    continue
                dest = last._successors[0].block
                if len(dest._nodes[0]._predecessors) != 1:
                    continue
                self.join_blocks(bb, dest)
                break
            else:
                break


def _load_param_nodes(cdfg: ControlDataFlowGraph,
                      xmlparams: lxml.etree.Element):

    for i, xmlparam in enumerate(xmlparams.iterchildren()):
        node = cdfg.new_node(
            ParameterNode,
            id_=xmlparam.get("id"),
            type_=xmlparam.get("type_"),
        )


def _load_asm_nodes(cdfg: ControlDataFlowGraph,
                    method: str,
                    xmlnodes: lxml.etree.Element):
    edges = []
    inputs = []

    for i, xmlnode in enumerate(xmlnodes.iterchildren()):
        bb_id = "{}/basic-blocks/{}".format(method, i)
        bb = cdfg.new_block(id_=bb_id)
        node = cdfg.new_node(
            ASMNode,
            id_=xmlnode.get("id"),
            block=bb,
            line=int(xmlnode.get("line", "-1")),
            opcode=int(xmlnode.get("opcode", "-1")),
        )
        xmlexits = xmlnode.find(ASM.exits)
        if xmlexits is not None:
            for exit in xmlexits:
                edges.append(
                    (node.unique_id, exit.get("to"))
                )
        xmlinputs = xmlnode.find(ASM.inputs)
        if xmlinputs is not None:
            for input_ in xmlinputs:
                if input_.tag == ASM("value-of"):
                    inputs.append(
                        (input_.get("from"), node.unique_id),
                    )
                else:
                    raise ValueError("unsupported input: {!r}".format(
                        input_.tag,
                    ))

    for from_, to in edges:
        from_node = cdfg.node_by_id(from_)
        to_node = cdfg.node_by_id(to)

        cdfg.add_successor(from_node, to_node)

    for from_, to in inputs:
        from_node = cdfg.node_by_id(from_)
        to_node = cdfg.node_by_id(to)

        cdfg.add_input(to_node, from_node)


def load_asm(tree: lxml.etree.Element) -> ControlDataFlowGraph:
    result = ControlDataFlowGraph()

    params = tree.find(ASM.parameters)
    if params is not None:
        _load_param_nodes(result, params)

    insns = tree.find(ASM.insns)
    if insns is not None:
        _load_asm_nodes(result, tree.get("id"), insns)

    result.simplify_basic_blocks()

    return result
