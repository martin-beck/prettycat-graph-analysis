#!/usr/bin/env python3
import itertools
import typing

import lxml.etree as etree


class Namespace:
    def __init__(self, ns):
        super().__init__()
        self.__dict__["namespace"] = ns

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return self(name)

    def __call__(self, name):
        return "{{{}}}{}".format(self.__dict__["namespace"], name)


IrGraph = Namespace("https://xmlns.zombofant.net/prettycat/1.0/ir-graph")


class Node:
    def __init__(self):
        super().__init__()
        self.slot = None
        self.node = None
        self.type_ = None
        self.inputs = []

    @classmethod
    def from_tree(cls, tree):
        instance = cls()
        instance.slot = tree.get("slot")
        instance.node = tree.get("node")
        instance.type_ = tree.get("type_")

        inputs = tree.find(IrGraph.inputs)

        if inputs is not None:
            instance.inputs = [
                node.get("object")
                for node in inputs.iterchildren(IrGraph.input)
            ]

        if tree.tag == IrGraph.call:
            instance.parameters = [
                (int(node.get("slot")), node.get("object"))
                for node in tree.iterchildren(IrGraph("pass"))
            ]
            instance.inputs = [
                object_
                for _, object_ in instance.parameters
            ]

        return instance


class NodeContainer:
    def __init__(self):
        super().__init__()
        self.nodes = []

    @classmethod
    def from_tree(cls, tree):
        instance = cls()
        instance.nodes = [
            Node.from_tree(node)
            for node in tree.iterchildren()
        ]
        return instance


class Block:
    def __init__(self):
        super().__init__()
        self.num = None
        self.nodes = None

    def joined_inputs(self) -> typing.Set[str]:
        result = set()
        for node in self.nodes.nodes:
            if node.inputs:
                result.update(node.inputs)
        return result

    @classmethod
    def from_tree(cls, tree):
        instance = cls()
        instance.num = int(tree.get("num"))
        instance.nodes = NodeContainer.from_tree(tree.find(IrGraph.nodes))
        instance.exits = [
            int(exit.get("block"))
            for exit in tree.find(IrGraph.exits).iterchildren(IrGraph.exit)
        ]
        return instance

    def __repr__(self):
        return "<{}.{} num={!r} #nodes={!r} at 0x{:x}>".format(
            type(self).__module__,
            type(self).__qualname__,
            self.num,
            len(self.nodes.nodes),
            id(self)
        )


class CDFG:
    def __init__(self):
        super().__init__()
        self.floating = None
        self.blocks = []

    def block_by_num(self, num: int) -> Block:
        for block in self.blocks:
            if block.num == num:
                return block
        raise KeyError(num)

    def node_by_slot(self, slot: str) -> Node:
        for node in itertools.chain(self.floating.nodes,
                                    *(block.nodes.nodes
                                      for block in self.blocks)):
            if node.slot == slot:
                return node
        raise KeyError(slot)

    def block_by_node(self, node: Node) -> Block:
        for block in self.blocks:
            if node in block.nodes.nodes:
                return block
        raise KeyError(node)

    @classmethod
    def from_tree(cls, tree):
        instance = cls()
        instance.floating = NodeContainer.from_tree(tree.find(IrGraph.floating))
        instance.blocks = [
            Block.from_tree(block)
            for block in tree.iterchildren(IrGraph.block)
        ]
        return instance

    def __repr__(self):
        return "<{}.{} #floating={!r} #blocks={!r} at 0x{:x}>".format(
            type(self).__module__,
            type(self).__qualname__,
            len(self.floating.nodes),
            len(self.blocks),
        )


class Method:
    def __init__(self, id_):
        super().__init__()
        self.id_ = id_
        self.parameters = []

    def get_class(self):
        return self.id_.rsplit(".", 1)[0]

    @classmethod
    def from_tree(cls, tree):
        instance = cls(tree.get("id"))
        for parameter in tree.find(IrGraph.parameters).iterchildren():
            if parameter.tag == IrGraph.this:
                parameter = (instance.id_ + "/param:0", instance.get_class())
            else:
                parameter = (parameter.get("slot"), parameter.get("type"))
            instance.parameters.append(parameter)
        instance.cdfg = CDFG.from_tree(tree.find(IrGraph.cdfg))
        return instance

    def __repr__(self):
        return "<{}.{} id_={!r} parameters={!r} at 0x{:x}>".format(
            type(self).__module__,
            type(self).__qualname__,
            self.id_,
            self.parameters,
            id(self)
        )


class Graph:
    def __init__(self):
        super().__init__()
        self.methods = {}

    def load(self, tree):
        methods = (
            Method.from_tree(method)
            for method in tree.iterchildren(IrGraph.method)
        )
        self.methods = {
            method.id_: method
            for method in methods
        }


def bb_graph_dot(args, graph):
    method = graph.methods[args.method]

    if args.outfile is None:
        f = sys.stdout
    else:
        f = open(args.outfile, "w")

    with f:
        print("digraph {", file=f)
        for floating in method.cdfg.floating.nodes:
            print('"{}" [label="{}"];'.format(floating.slot, floating.node), file=f)
            for input_ in floating.inputs:
                try:
                    node = method.cdfg.node_by_slot(input_)
                except KeyError:
                    continue
                try:
                    src_block = method.cdfg.block_by_node(node)
                except KeyError:
                    print(
                        '"{}" -> "{}" [style=dashed];'.format(
                            node.slot,
                            floating.slot,
                        ),
                        file=f
                    )
                else:
                    print(
                        'bb_{} -> "{}" [style=dashed];'.format(
                            src_block.num,
                            floating.slot
                        ),
                        file=f
                    )
        for block in method.cdfg.blocks:
            print("bb_{};".format(block.num), file=f)
            for exit in block.exits:
                print("bb_{} -> bb_{};".format(block.num, exit), file=f)
            for input_ in block.joined_inputs():
                try:
                    node = method.cdfg.node_by_slot(input_)
                except KeyError:
                    continue
                try:
                    src_block = method.cdfg.block_by_node(node)
                except KeyError:
                    print(
                        '"{}" -> bb_{} [style=dashed];'.format(
                            node.slot,
                            block.num,
                        ),
                        file=f
                    )
                else:
                    print(
                        "bb_{} -> bb_{} [style=dashed];".format(
                            src_block.num,
                            block.num,
                        ),
                        file=f
                    )
        print("}", file=f)


def cdfg_dot(args, graph):
    method = graph.methods[args.method]

    if args.outfile is None:
        f = sys.stdout
    else:
        f = open(args.outfile, "w")

    def emit_inputs(node, f):
        for i, input_ in enumerate(node.inputs):
            try:
                src_node = method.cdfg.node_by_slot(input_)
            except KeyError:
                # FIXME: parameters don’t have nodes
                src_node = input_
            else:
                src_node = src_node.node

            print(
                '"{}" -> "{}" [style=dashed,headlabel="in {}"]'.format(
                    src_node,
                    node.node,
                    i,
                ),
                file=f
            )

    with f:
        print('digraph {', file=f)

        for floating in method.cdfg.floating.nodes:
            print(
                '"{}" [label="{}"];'.format(
                    floating.node,
                    floating.node,
                ),
                file=f
            )
            emit_inputs(floating, f)

        for block in method.cdfg.blocks:
            prev = None
            for node in block.nodes.nodes:
                print(
                    '"{}" [label="{}"]'.format(
                        node.node,
                        node.node,
                    ),
                    file=f
                )

                if prev is not None:
                    print(
                        '"{}" -> "{}";'.format(
                            prev.node,
                            node.node,
                        ),
                        file=f
                    )

                emit_inputs(node, f)

                prev = node

            for i, exit in enumerate(block.exits):
                target_block = method.cdfg.block_by_num(exit)
                src_node = block.nodes.nodes[-1]
                dest_node = target_block.nodes.nodes[0]

                print(
                    '"{}" -> "{}" [taillabel="branch {}"];'.format(
                        src_node.node,
                        dest_node.node,
                        i,
                    ),
                    file=f
                )

        print('}', file=f)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph",
        type=argparse.FileType("rb")
    )

    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("bb-graph-dot")
    subparser.set_defaults(func=bb_graph_dot)
    subparser.add_argument(
        "method",
        help="Method to dump the graph for"
    )
    subparser.add_argument(
        "outfile",
        nargs="?",
        help="File to dump the graph to"
    )

    subparser = subparsers.add_parser("cdfg-dot")
    subparser.set_defaults(func=cdfg_dot)
    subparser.add_argument(
        "method",
        help="Method to dump the graph for"
    )
    subparser.add_argument(
        "outfile",
        nargs="?",
        help="File to dump the graph to"
    )

    args = parser.parse_args()

    if not hasattr(args, "func"):
        print("a subcommand is required", file=sys.stderr)
        parser.print_help()
        return 1

    with args.graph as f:
        data = etree.parse(f)

    graph = Graph()
    graph.load(data.getroot())
    args.func(args, graph)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
