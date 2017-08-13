#!/usr/bin/env python3
import itertools
import typing

import prettycdfg.nodes

from prettycdfg.xmlutil import ASM

import lxml.etree as etree


def load_graphs(f):
    with f:
        tree = etree.parse(f)

    methods = {}
    if tree.getroot().tag == ASM.asm:
        loader = prettycdfg.nodes.load_asm
    else:
        raise ValueError("unsupported graph type: {!r}".format(
            tree.getroot().tag
        ))

    for xmlmethod in tree.getroot():
        id_ = xmlmethod.get("id")
        methods[id_] = loader(xmlmethod)

    return methods


def bb_graph_dot(args,
                 graphs: typing.Mapping[
                     str,
                     prettycdfg.nodes.ControlDataFlowGraph]):
    cdfg = graphs[args.method]

    if args.outfile is None:
        f = sys.stdout
    else:
        f = open(args.outfile, "w")

    def joined_inputs(block):
        for node in block.nodes:
            yield from node.inputs

    with f:
        print("digraph {", file=f)
        for floating in cdfg.floating_nodes:
            print(
                '"{}" [label="{}"];'.format(
                    floating.unique_id,
                    floating.unique_id,
                ),
                file=f
            )
            for input_ in floating.inputs:
                try:
                    node = cdfg.node_by_id(input_)
                except KeyError:
                    continue
                try:
                    src_block = cdfg.block_by_node(node)
                except KeyError:
                    print(
                        '"{}" -> "{}" [style=dashed];'.format(
                            node.unique_id,
                            floating.unique_id,
                        ),
                        file=f
                    )
                else:
                    print(
                        'bb_{} -> "{}" [style=dashed];'.format(
                            src_block.unique_id,
                            floating.unique_id
                        ),
                        file=f
                    )

        for block in cdfg.blocks:
            print('"bb_{}";'.format(block.unique_id), file=f)
            for exit in block.exits:
                print('"bb_{}" -> "bb_{}";'.format(block.unique_id,
                                                   exit.unique_id), file=f)
            for node in joined_inputs(block):
                try:
                    src_block = cdfg.block_by_node(node)
                except KeyError:
                    print(
                        '"{}" -> "bb_{}" [style=dashed];'.format(
                            node.unique_id,
                            block.unique_id,
                        ),
                        file=f
                    )
                else:
                    print(
                        '"bb_{}" -> "bb_{}" [style=dashed];'.format(
                            src_block.unique_id,
                            block.unique_id,
                        ),
                        file=f
                    )
        print("}", file=f)


def cdfg_dot(args,
             graphs: typing.Mapping[
                 str,
                 prettycdfg.nodes.ControlDataFlowGraph]):
    cdfg = graphs[args.method]

    if args.outfile is None:
        f = sys.stdout
    else:
        f = open(args.outfile, "w")

    def emit_inputs(node, f):
        for i, input_ in enumerate(node.inputs):
            print(
                '"{}" -> "{}" [style=dashed,headlabel="in {}"]'.format(
                    input_.unique_id,
                    node.unique_id,
                    i,
                ),
                file=f
            )

    with f:
        print('digraph {', file=f)
        if args.group_basic_blocks:
            print('compound=true;', file=f)

        for floating in cdfg.floating_nodes:
            print(
                '"{}" [label="{}"];'.format(
                    floating.unique_id,
                    floating.unique_id,
                ),
                file=f
            )
            emit_inputs(floating, f)

        for block in cdfg.blocks:
            prev = None
            if args.group_basic_blocks:
                print('subgraph "cluster_bb_{}" {{'.format(block.unique_id), file=f)
                print('label="bb {}";'.format(block.unique_id), file=f)
                print('color=black;', file=f)
            for node in block.nodes:
                print(
                    '"{}" [label="{}"]'.format(
                        node.unique_id,
                        node.local_id,
                    ),
                    file=f
                )

                if prev is not None:
                    print(
                        '"{}" -> "{}";'.format(
                            prev.unique_id,
                            node.unique_id,
                        ),
                        file=f
                    )

                emit_inputs(node, f)

                prev = node

            if args.group_basic_blocks:
                print('}', file=f)

            for i, target_block in enumerate(block.exits):
                src_node = list(block.nodes)[-1]
                dest_node = list(target_block.nodes)[0]

                cluster_attrs = ""
                if args.group_basic_blocks:
                    cluster_attrs = ',lhead="cluster_bb_{}",ltail="cluster_bb_{}"'.format(
                        target_block.unique_id,
                        block.unique_id,
                    )

                print(
                    '"{}" -> "{}" [taillabel="branch {}"{}];'.format(
                        src_node.unique_id,
                        dest_node.unique_id,
                        i,
                        cluster_attrs,
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
    subparser.add_argument(
        "-g", "--group-basic-blocks",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if not hasattr(args, "func"):
        print("a subcommand is required", file=sys.stderr)
        parser.print_help()
        return 1

    methods = load_graphs(args.graph)
    args.func(args, methods)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
