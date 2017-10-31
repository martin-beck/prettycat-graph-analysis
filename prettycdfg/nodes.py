import abc
import copy
import enum
import itertools
import uuid
import typing

import lxml.etree

from .opcodes import Opcode
from .xmlutil import ASM


class EdgeType(enum.Enum):
    DATA_FLOW = 'data'
    CONTROL_FLOW = 'control'


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

    def __repr__(self):
        return "<{}.{} {!r}>".format(
            __name__,
            type(self).__qualname__,
            self._unique_id,
        )


class Node(AbstractNode):
    def __init__(self, unique_id):
        super().__init__(unique_id)
        self._cf_in = []
        self._cf_out = []
        self._df_in = []
        self._df_out = []
        self._block = None

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def inputs(self) -> typing.Iterable['Node']:
        """
        A list of unique IDs of input nodes.
        """
        return (edge.from_ for edge in self._df_in)

    @property
    def successors(self) -> typing.Iterable['Node']:
        return (edge.to for edge in self._cf_out)

    @property
    def predecessors(self) -> typing.Iterable['Node']:
        return (edge.from_ for edge in self._cf_in)

    @property
    def block(self) -> 'BasicBlock':
        return self._block

    def __copy__(self):
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__.copy())
        result._cf_in = []
        result._cf_out = []
        result._df_in = []
        result._df_out = []
        result._block = None
        result._unique_id = None
        return result

    def __str__(self):
        return "{}:unique_id={!r}".format(
            type(self).__qualname__,
            self._unique_id,
        )

    @property
    def is_return(self) -> bool:
        return False

    @property
    def is_parameter(self) -> bool:
        return False


class ASMNode(Node):
    RETURN_OPCODES = [Opcode.IRETURN, Opcode.ARETURN, Opcode.LRETURN,
                      Opcode.FRETURN, Opcode.DRETURN, Opcode.RETURN]

    def __init__(self, unique_id,
                 line: typing.Optional[int] = None,
                 opcode: typing.Optional[int] = None,
                 call_target: typing.Optional[str] = None):
        super().__init__(unique_id)
        self._line = line
        self._opcode = opcode if opcode is None else Opcode(opcode)
        self._call_target = call_target

    @property
    def call_target(self) -> str:
        return self._call_target

    @property
    def line(self) -> int:
        return self._line

    @property
    def opcode(self) -> Opcode:
        return self._opcode

    @property
    def local_id(self) -> str:
        return self.unique_id.split("/", 1)[1]

    @property
    def is_return(self) -> bool:
        return self._opcode in self.RETURN_OPCODES

    def __str__(self):
        parts = []
        if self._line is not None:
            parts.append("line={}".format(self._line))
        if self._opcode is not None:
            parts.append("opcode={!r}".format(self._opcode))
        if self._call_target is not None:
            parts.append("call_target={!r}".format(self._call_target))
        return "ASM:{}".format(";".join(parts))

    def __repr__(self):
        return "<{}.{} line={!r} opcode={!r} call_target{!r}>".format(
            __name__,
            type(self).__qualname__,
            self._line,
            self._opcode,
            self._call_target,
        )


class MergeNode(Node):
    def __init__(self, unique_id):
        super().__init__(unique_id)


class ExceptionNode(Node):
    def __init__(self, unique_id, type_: typing.Optional[str] = None):
        super().__init__(unique_id)
        self._type = type_

    @property
    def type_(self):
        return self._type

    def __str__(self):
        return "Exc:unique_id={!r};type={!r}".format(
            self._unique_id,
            self._type,
        )


class ParameterNode(Node):
    def __init__(self, unique_id,
                 type_: typing.Optional[str] = None):
        super().__init__(unique_id)
        self._type = type_

    @property
    def type_(self):
        return self._type

    def __str__(self):
        return "Param:unique_id={!r};type={!r}".format(
            self._unique_id,
            self._type,
        )

    @property
    def is_parameter(self):
        return True


class InlineNode(Node):
    def __init__(self, unique_id, inlined_id: str):
        super().__init__(unique_id)
        self._inlined_id = inlined_id

    @property
    def inlined_id(self) -> str:
        return self._inlined_id


class PreInlineNode(InlineNode):
    def __str__(self):
        return "PreInline:unique_id={!r};inlined_id={!r}".format(
            self._unique_id,
            self._inlined_id,
        )


class PostInlineNode(InlineNode):
    def __str__(self):
        return "PostInline:unique_id={!r};inlined_id={!r}".format(
            self._unique_id,
            self._inlined_id,
        )


class AbstractEdge:
    def __init__(self, type_: EdgeType,
                 from_: AbstractNode,
                 to: AbstractNode, **kwargs):
        super().__init__(**kwargs)
        self._type = type_
        self._from = from_
        self._to = to

    @property
    def type_(self) -> EdgeType:
        return self._type

    @property
    def from_(self) -> AbstractNode:
        return self._from

    @property
    def to(self) -> AbstractNode:
        return self._to

    def rebind(self, *,
               to: AbstractNode = None,
               from_: AbstractNode = None) -> "AbstractEdge":
        new_instance = copy.copy(self)
        if to is not None:
            new_instance._to = to
        if from_ is not None:
            new_instance._from = from_
        return new_instance

    def __repr__(self):
        return "<{}.{} {!r} to {!r}>".format(
            type(self).__module__,
            type(self).__qualname__,
            self._from,
            self._to,
        )


class DataFlowEdge(AbstractEdge):
    def __init__(self, from_: AbstractNode, to: AbstractNode, **kwargs):
        super().__init__(EdgeType.DATA_FLOW, from_, to, **kwargs)


class ControlFlowEdge(AbstractEdge):
    def __init__(self, from_: AbstractNode, to: AbstractNode, *,
                 is_exceptional: bool = False, **kwargs):
        super().__init__(EdgeType.CONTROL_FLOW, from_, to)
        self._is_exceptional = is_exceptional

    @property
    def is_exceptional(self) -> bool:
        return self._is_exceptional


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

    def assert_consistency(self):
        """
        :raises AssertionError: if internal consistency is off
        """
        seen_nodes = set()
        for node in self._floating_nodes:
            assert node not in seen_nodes
            seen_nodes.add(node)

        for block in self._blocks:
            prev_node = None
            for node in block.nodes:
                assert node not in seen_nodes
                seen_nodes.add(node)
                if prev_node is not None:
                    assert prev_node._successors == [node]
                    assert node._predecessors == [prev_node]
                assert node._block is block, (node, block)

        seen_in_edges = set()
        seen_out_edges = set()

        def check_in_edges(node, type_):
            for in_edge in getattr(node, "_{}_in".format(type_)):
                assert in_edge not in seen_in_edges, \
                    "in_edge referenced by multiple nodes"
                seen_in_edges.add(in_edge)
                assert in_edge.to is node, \
                    "holder of in_edge is not the actual destination"
                assert in_edge in getattr(
                    in_edge.from_, "_{}_out".format(type_)), \
                    "in_edge not held by origin"

        def check_out_edges(node, type_):
            for out_edge in getattr(node, "_{}_out".format(type_)):
                assert out_edge not in seen_out_edges, \
                    "out_edge referenced by multiple nodes"
                seen_out_edges.add(out_edge)
                assert out_edge.from_ is node, \
                    "holder of out_edge is not the actual origin"
                assert out_edge in getattr(
                    out_edge.to, "_{}_in".format(type_)), \
                    "out_edge not held by destination"

        for node in self.nodes:
            check_in_edges(node, "cf")
            check_in_edges(node, "df")
            check_out_edges(node, "cf")
            check_out_edges(node, "df")

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

    @property
    def parameters(self) -> typing.Iterable[Node]:
        """
        An iterable of nodes representing inputs to the control/data flow
        graph.

        The iterable is sorted in the order of parameters.
        """
        return (
            node for node in self.nodes
            if node.is_parameter
        )

    @property
    def returns(self) -> typing.Iterable[Node]:
        return (
            node for node in self.nodes
            if node.is_return
        )

    @property
    def heads(self) -> typing.Iterable[BasicBlock]:
        """
        An iterable of basic blocks whose first node does not have a
        predecessor.
        """
        return (
            block for block in self._blocks
            if block._nodes and not block._nodes[0]._cf_in
        )

    @property
    def tails(self) -> typing.Iterable[BasicBlock]:
        """
        An iterable of basic blocks whose last node does not have a successor.
        """
        return (
            block for block in self._blocks
            if block._nodes and not block._nodes[-1]._cf_out
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
                old_last = block._nodes[-1]
                in_edge = ControlFlowEdge(old_last, node)
                for old_out_edge in old_last._cf_out:
                    new_out_edge = old_out_edge.rebind(
                        from_=node,
                    )
                    node._cf_out.append(new_out_edge)
                    old_out_edge.to._cf_in[
                        old_out_edge.to._cf_in.index(old_out_edge)
                    ] = new_out_edge
                node._cf_in.append(in_edge)
                old_last._cf_out[:] = [in_edge]
            block._nodes.append(node)
        else:
            self._floating_nodes.append(node)
        return node

    def move_node(self, node: Node, insert_before: Node):
        """
        Move a node before another node.

        :raises ValueError: if `insert_before` is not part of any basic block.
        :raises ValueError: if `node` has multiple predecessors but not exactly
            one successor.
        :raises ValueError: if `node` has multiple successors bot not exactly
            one predecessor.
        """
        before_edges = list(insert_before._cf_in)

        if node.block is not None:
            self.detach_node(node)

        # now we patch the destination links

        for in_edge in insert_before._cf_in:
            new_edge = in_edge.rebind(to=node)
            node._cf_in.append(new_edge)
            in_edge.to._cf_in.remove(in_edge)
            in_edge.from_._cf_out[in_edge.from_._cf_out.index(in_edge)] = \
                new_edge
        insert_before._cf_in.clear()
        self._floating_nodes.remove(node)
        node._block = insert_before._block
        insert_before._block._nodes.insert(
            insert_before._block._nodes.index(insert_before),
            node,
        )

        new_edge = ControlFlowEdge(node, insert_before)
        node._cf_out.append(new_edge)
        insert_before._cf_in.append(new_edge)

    def detach_node(self, node: Node):
        """
        Make a node floating.

        :raises ValueError: if `node` has multiple predecessors and multiple
            successors
        :raises ValueError: if `node` is floating

        .. seealso::

            :meth:`move_node`
                can be used to put a node into a basic block.
        """
        if node.block is None:
            raise ValueError("cannot detach floating node")

        old_outbound = list(node._cf_out)
        old_inbound = list(node._cf_in)

        if len(old_inbound) > 1 and len(old_outbound) > 1:
            raise ValueError("cannot detach node with multiple predecessors "
                             "and multiple successors")

        # we first patch the old links

        new_edges = [
            (in_edge, out_edge, in_edge.rebind(to=out_edge.to))
            for in_edge in old_inbound
            for out_edge in old_outbound
        ]

        out_edge_map = {
            (out_edge, in_edge.from_): new_edge
            for in_edge, out_edge, new_edge in new_edges
        }

        in_edge_map = {
            (in_edge, out_edge.to): new_edge
            for in_edge, out_edge, new_edge in new_edges
        }

        for out_edge in old_outbound:
            out_edge.to._cf_in.remove(out_edge)
            out_edge.to._cf_in.extend(
                in_edge_map[in_edge, out_edge.to]
                for in_edge in old_inbound
            )

        for in_edge in old_inbound:
            in_edge.from_._cf_out.remove(in_edge)
            in_edge.from_._cf_out.extend(
                out_edge_map[out_edge, in_edge.from_]
                for out_edge in old_outbound
            )

        node._cf_in.clear()
        node._cf_out.clear()
        node._block._nodes.remove(node)
        self._floating_nodes.append(node)
        node._block = None

    def add_successor(self, at: Node, successor: Node, **kwargs):
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

        new_edge = ControlFlowEdge(at, successor, **kwargs)
        at._cf_out.append(new_edge)
        successor._cf_in.append(new_edge)

    def remove_successor(self, at: Node, successor: Node):
        """
        Remove a successor from a node.

        :raises ValueError: if `at` and `successor` are in the same basic block
        """
        if at.block is successor.block:
            raise ValueError(
                "cannot remove successors within basic block; use split_block "
                "instead"
            )

        for i, out_edge in enumerate(at._cf_out):
            if out_edge.to is successor:
                break
        else:
            raise ValueError(
                "{!r} is not a successor of {!r}".format(
                    successor,
                    at,
                )
            ) from None

        del at._cf_out[i]
        successor._cf_in.remove(out_edge)

    def add_input(self, at: Node, source: Node):
        """
        Add an input to a node.

        .. note::

            This does not check that the data flow graph is still valid.
        """
        edge = DataFlowEdge(source, at)
        source._df_out.append(edge)
        at._df_in.append(edge)

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
        old_block = at_node.block
        old_index = old_block._nodes.index(at_node)
        if old_index == 0:
            raise ValueError(
                "cannot split a block at its first node"
            )
        block = self.new_block()
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
        if b1._nodes and len(b1._nodes[-1]._cf_out) > 1:
            raise ValueError("b1 has multiple successors")
        if b2._nodes and len(b2._nodes[0]._cf_in) > 1:
            raise ValueError("b2 has multiple predecessors")
        b1._nodes.extend(b2._nodes)
        for node in b2._nodes:
            node._block = b1
        b2._nodes.clear()
        self._blocks.remove(b2)
        return b1

    def merge(self, other_cdfg: 'ControlDataFlowGraph'):
        """
        Merge another graph into this graph.

        The blocks and nodes from `other_cdfg` are *copied* and added to this
        graph.
        """
        blockmap = {
            old_block: self.new_block()
            for old_block in other_cdfg.blocks
        }
        blockmap[None] = None

        nodemap = {}

        new_nodes = []
        new_blocks = [blockmap[old_block] for old_block in other_cdfg.blocks]

        control_flow = []
        data_flow = []

        for old_node in other_cdfg.nodes:
            block = blockmap[old_node.block]
            new_node = copy.copy(old_node)
            new_node._cf_in.clear()
            new_node._df_in.clear()
            new_node._cf_out.clear()
            new_node._df_out.clear()

            def fake_new_node(unique_id):
                new_node._unique_id = unique_id
                return new_node

            nodemap[old_node] = self.new_node(fake_new_node, block=block)

            for out_edge in old_node._cf_out:
                if out_edge.to.block == old_node.block:
                    continue
                control_flow.append(out_edge)

            for in_edge in old_node._df_in:
                data_flow.append(in_edge)

            new_nodes.append(new_node)

        for out_edge in control_flow:
            new_edge = out_edge.rebind(
                from_=nodemap[out_edge.from_],
                to=nodemap[out_edge.to],
            )
            new_edge.from_._cf_out.append(new_edge)
            new_edge.to._cf_in.append(new_edge)

        for in_edge in data_flow:
            new_edge = in_edge.rebind(
                from_=nodemap[in_edge.from_],
                to=nodemap[in_edge.to],
            )
            new_edge.from_._df_out.append(new_edge)
            new_edge.to._df_in.append(new_edge)

        return new_nodes, new_blocks

    def inline_call(self, call_node: Node,
                    method_cdfg: 'ControlDataFlowGraph',
                    inlined_id: typing.Optional[str] = None):
        """
        Inline a given call node.

        :param call_node: The call to inline.
        :type call_node: :class:`CallNode`
        :param method_cdfg: The control/data flow graph of the called method.
        :type method_cdfg: :class:`ControlDataFlowGraph`

        The control/data flow graph of the inlined method will be merged into
        this control/data flow graph. The unique IDs are relabeled to allow
        to inline the same method at different callsites.

        The relabeling uses a new UUID for each inlining. It is the callers
        responsibility to avoid recursive inlining.
        """

        # TODO: parameters
        # TODO: return values
        # TODO: exception branches

        inlined_id = inlined_id or ("urn:uuid:" + str(uuid.uuid4()))

        try:
            head, = method_cdfg.heads
        except ValueError:
            raise ValueError(
                "cannot inline graph with not exactly one head block"
            )
        tails = list(method_cdfg.tails)
        if not tails:
            raise ValueError(
                "cannot inline graph with no tails"
            )

        inputs = list(call_node._df_in)
        if len(inputs) != len(list(method_cdfg.parameters)):
            raise ValueError(
                "number of inputs and number of parameters mismatch"
            )

        pre_inline = self.new_node(PreInlineNode,
                                   inlined_id=inlined_id)
        self.move_node(pre_inline, call_node)

        post_inline = self.new_node(PostInlineNode,
                                    inlined_id=inlined_id)
        self.move_node(post_inline, call_node)

        prev_block = post_inline.block
        next_block = self.split_block(post_inline)

        new_nodes, new_blocks = self.merge(method_cdfg)
        nodemap = dict(zip(method_cdfg.nodes, new_nodes))

        self.remove_successor(pre_inline, post_inline)

        self.add_successor(pre_inline, nodemap[head._nodes[0]])
        for tail in tails:
            self.add_successor(nodemap[tail._nodes[-1]], post_inline)

        parameters = [
            nodemap[param]
            for param in method_cdfg.parameters
        ]

        parametermap = {
            param_node: in_edge
            for param_node, in_edge in zip(parameters, inputs)
        }

        for node in nodemap.values():
            for i, in_edge in enumerate(list(node._df_in)):
                try:
                    replacement = parametermap[in_edge.from_]
                except KeyError:
                    continue
                # the old edge will be deleted by remove_node(parameter) later
                new_edge = in_edge.rebind(
                    from_=replacement.from_,
                )
                new_edge.from_._df_out.append(new_edge)
                node._df_in.append(new_edge)

        for parameter in parameters:
            self.remove_node(parameter)

        returns = [
            nodemap[return_node]
            for return_node in method_cdfg.returns
        ]
        if len(returns) != 1:
            return_value_node = self.new_node(MergeNode)
            for return_node in returns:
                self.add_input(return_value_node, return_node)
        else:
            return_value_node, = returns

        for out_edge in call_node._df_out:
            new_edge = out_edge.rebind(from_=return_value_node)
            out_edge.to._df_in[out_edge.to._df_in.index(out_edge)] = \
                new_edge
            new_edge.from_._df_out.append(new_edge)

        call_node._df_out.clear()

        self.remove_node(call_node)

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
                if len(last._cf_out) != 1:
                    continue
                dest = last._cf_out[0].to.block
                if len(dest._nodes[0]._cf_in) != 1:
                    continue
                self.join_blocks(bb, dest)
                break
            else:
                break

    def remove_node(self, node: Node):
        """
        Remove the node from the CDFG.

        :param node: The node to remove.
        :type node: :class:`Node`
        """
        block = node._block
        if block is not None:
            self.detach_node(node)
        self._floating_nodes.remove(node)
        for in_edge in node._df_in:
            in_edge.from_._df_out.remove(in_edge)
        for out_edge in node._df_out:
            out_edge.to._df_in.remove(out_edge)
        node._df_in.clear()
        node._df_out.clear()


def _load_param_nodes(cdfg: ControlDataFlowGraph,
                      xmlparams: lxml.etree.Element):

    for i, xmlparam in enumerate(xmlparams.iterchildren()):
        node = cdfg.new_node(
            ParameterNode,
            id_=xmlparam.get("id"),
            type_=xmlparam.get("type"),
        )


def _load_asm_inputs(cdfg: ControlDataFlowGraph,
                     dest_unique_id: str,
                     xmlinputs: lxml.etree.Element):
    for input_ in xmlinputs:
        if input_.tag == ASM("value-of"):
            yield (input_.get("from"), dest_unique_id)
        elif input_.tag == ASM("merge"):
            merge_node = cdfg.new_node(
                MergeNode,
            )
            yield (merge_node.unique_id, dest_unique_id)
            yield from _load_asm_inputs(cdfg, merge_node.unique_id, input_)
        elif input_.tag == ASM("exception"):
            exception_node = cdfg.new_node(
                ExceptionNode,
                type_=input_.get("type")
            )
            yield (exception_node.unique_id, dest_unique_id)
        else:
            raise ValueError("unsupported input: {!r}".format(
                input_.tag,
            ))


def _load_asm_nodes(cdfg: ControlDataFlowGraph,
                    method: str,
                    xmlnodes: lxml.etree.Element):
    edges = []
    inputs = []

    for i, xmlnode in enumerate(xmlnodes.iterchildren()):
        bb_id = "{}/basic-blocks/{}".format(method, i)
        bb = cdfg.new_block(id_=bb_id)

        line = xmlnode.get("line")
        if line is not None:
            line = int(line)

        opcode = xmlnode.get("opcode")
        if opcode is not None:
            opcode = int(opcode)

        xmlcall_target = xmlnode.find(ASM("call-target"))
        if xmlcall_target is not None:
            call_target = xmlcall_target.get("target")
        else:
            call_target = None

        node = cdfg.new_node(
            ASMNode,
            id_=xmlnode.get("id"),
            block=bb,
            line=line,
            opcode=opcode,
            call_target=call_target,
        )

        xmlexits = xmlnode.find(ASM.exits)
        if xmlexits is not None:
            for exit in xmlexits:
                edges.append(
                    (node.unique_id, exit.get("to"))
                )
        xmlinputs = xmlnode.find(ASM.inputs)
        if xmlinputs is not None:
            inputs.extend(_load_asm_inputs(cdfg, node.unique_id, xmlinputs))

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

    for node in list(result.nodes):
        if node.block is None:
            continue
        if not isinstance(node, ASMNode):
            continue
        if node.opcode == -1:
            try:
                result.detach_node(node)
                result.remove_node(node)
            except ValueError:
                pass

    return result
