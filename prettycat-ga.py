#!/usr/bin/env python3
import collections.abc
import copy
import fnmatch
import functools
import itertools
import logging
import typing

import prettycdfg.nodes
import prettycdfg.opcodes

from prettycdfg.xmlutil import ASM

import lxml.etree as etree


class LazyGraphs(collections.abc.Mapping):
    def __init__(self, tree, loader):
        super().__init__()
        self._loader = loader
        self._treeindex = {
            xmlmethod.get("id"): xmlmethod
            for xmlmethod in tree.getroot()
        }
        self._loaded = {}
        self._filters = []

    def add_filter(self, filter_func):
        self._filters.append(filter_func)

    def _load(self, key):
        # print("\nLOADING", key, "\n")
        cdfg = self._loader(self._treeindex[key])
        for filter_func in self._filters:
            filter_func(cdfg)
        self._loaded[key] = cdfg
        return cdfg

    def __getitem__(self, key):
        try:
            return self._loaded[key]
        except KeyError:
            pass
        return self._load(key)

    def __iter__(self):
        return iter(self._treeindex)

    def __len__(self):
        return len(self._treeindex)

    def __contains__(self, key):
        return key in self._treeindex


def load_type_overrides(f) -> typing.Mapping[str, str]:
    mapping = {}

    for line in f:
        if line.startswith("#"):
            continue

        if not line.strip():
            continue

        try:
            lhs, rhs = line.split()
        except ValueError:
            raise ValueError("malformed type override line: {!r}".format(
                line
            ))

        mapping[lhs] = rhs

    return mapping


def load_method_matchers(f) -> typing.Sequence[typing.Callable]:
    matchers = []

    for line in f:
        if line.startswith("#"):
            continue

        if not line.strip():
            continue
        line = line[:-1]

        if any(ch == "*" or ch == "?" for ch in line):
            matchers.append(
                functools.partial(fnmatch.fnmatch, pat=line)
            )
        else:
            matchers.append(
                lambda x: x == line
            )

    return matchers


def load_graphs(f):
    with f:
        tree = etree.parse(f)

    methods = {}
    if tree.getroot().tag == ASM.asm:
        return LazyGraphs(tree, prettycdfg.nodes.load_asm)
    elif tree.getroot().tag == ASM.graph:
        return {
            tree.getroot().get("id"): prettycdfg.nodes.load_asm(tree.getroot())
        }
    else:
        raise ValueError("unsupported graph type: {!r}".format(
            tree.getroot().tag
        ))


def bb_graph_dot(args,
                 graphs: typing.Mapping[
                     str,
                     prettycdfg.nodes.ControlDataFlowGraph]):
    cdfg = get_full_graph(args, graphs)

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
                        '"{}" -> "bb_{}" [style=dashed,color=blue];'.format(
                            node.unique_id,
                            block.unique_id,
                        ),
                        file=f
                    )
                else:
                    print(
                        '"bb_{}" -> "bb_{}" [style=dashed,color=blue];'.format(
                            src_block.unique_id,
                            block.unique_id,
                        ),
                        file=f
                    )
        print("}", file=f)


def inline_call(
        cdfg: prettycdfg.nodes.ControlDataFlowGraph,
        node: prettycdfg.nodes.Node,
        graphs: typing.Mapping[
            str,
            prettycdfg.nodes.ControlDataFlowGraph],
        seen: typing.Set[str]):
    logger = logging.getLogger("inline")

    if node.call_target in seen:
        logger.warning("inline aborted due to recursion at %r",
                       node.call_target)
        return

    seen = seen | {node.call_target}

    try:
        src_graph = graphs[node.call_target]
    except KeyError:
        logger.warning("cannot inline %r: graph not available",
                       node.call_target)
        return

    new_graph = prettycdfg.nodes.ControlDataFlowGraph()
    new_graph.merge(src_graph)
    del src_graph

    for child_node in list(new_graph.nodes):
        if (not hasattr(child_node, "call_target") or
                child_node.call_target is None):
            continue
        logger.debug("inlining %r into %r",
                     child_node.call_target,
                     node.call_target)
        inline_call(new_graph, child_node, graphs, seen)

    cdfg.inline_call(node, new_graph, inlined_id=node.call_target)
    cdfg.assert_consistency()


def inline_calls(
        id_: str,
        cdfg: prettycdfg.nodes.ControlDataFlowGraph,
        graphs: typing.Mapping[
            str,
            prettycdfg.nodes.ControlDataFlowGraph]):
    # print()
    # print("--- INLINE START ---")
    # print()
    for node in list(cdfg.nodes):
        if not hasattr(node, "call_target") or node.call_target is None:
            continue
        # print("inlining {} into {}".format(node.call_target, id_))
        inline_call(cdfg, node, graphs, {id_})
    cdfg.assert_consistency()


def strip_asm_stack_instructions(cdfg: prettycdfg.nodes.ControlDataFlowGraph):
    TO_DELETE = [
        prettycdfg.opcodes.Opcode.ILOAD,
        prettycdfg.opcodes.Opcode.LLOAD,
        prettycdfg.opcodes.Opcode.FLOAD,
        prettycdfg.opcodes.Opcode.DLOAD,
        prettycdfg.opcodes.Opcode.ALOAD,
        prettycdfg.opcodes.Opcode.ISTORE,
        prettycdfg.opcodes.Opcode.LSTORE,
        prettycdfg.opcodes.Opcode.FSTORE,
        prettycdfg.opcodes.Opcode.DSTORE,
        prettycdfg.opcodes.Opcode.ASTORE,
        prettycdfg.opcodes.Opcode.DUP,
        prettycdfg.opcodes.Opcode.RETURN,
        prettycdfg.opcodes.Opcode.INVALID,
    ]

    for node in list(cdfg.nodes):
        if not isinstance(node, prettycdfg.nodes.ASMNode):
            continue
        if node.opcode in TO_DELETE:
            if (not node._df_in and
                    not node._df_out and
                    len(node._cf_in) <= 1 and
                    len(node._cf_out) <= 1):
                cdfg.remove_node(node)
                cdfg.assert_consistency()


def apply_type_overrides(
        overrides: typing.Mapping[str, str],
        graph: prettycdfg.nodes.ControlDataFlowGraph):
    logger = logging.getLogger("type_overrides")

    for node in graph.nodes:
        if not isinstance(node, prettycdfg.nodes.ASMNode):
            continue

        if node.opcode not in prettycdfg.opcodes.CALL_OPCODES:
            continue
        if node.opcode == prettycdfg.opcodes.Opcode.INVOKESTATIC:
            continue

        old_call_target = node.call_target
        assert old_call_target.startswith("java:")
        old_owner, method_sig = old_call_target[5:].rsplit(".", 1)
        assert "[" not in old_owner, \
            "splitting doesn’t work, we need something stronger"

        try:
            new_owner = overrides[old_owner]
        except KeyError:
            # print("no type override for {}".format(node),
            #       file=sys.stderr)
            continue

        # patch things!
        node._call_target = "java:{}.{}".format(new_owner, method_sig)

        logger.info(
            "patched %r -> %r",
            old_call_target,
            node.call_target,
        )


def strip_non_calls(
        method_matchers: typing.Sequence[typing.Callable],
        graph: prettycdfg.nodes.ControlDataFlowGraph):
    logger = logging.getLogger("strip_non_calls")

    def try_remove_node(cdfg, node):
        logger.debug("removing node: %s", node)
        try:
            cdfg.remove_node(node)
        except ValueError as exc:
            logger.info("cannot strip non-call node %r: %s", node, exc)
            return
        # logging.debug("checking consistency")
        # graph.assert_consistency()

    for node in list(graph.nodes):
        if not node.block:
            continue

        if not isinstance(node, prettycdfg.nodes.ASMNode):
            try_remove_node(graph, node)
            continue

        if node.opcode not in prettycdfg.opcodes.CALL_OPCODES:
            try_remove_node(graph, node)
            continue

        assert hasattr(node, "call_target") and node.call_target

        # print(node.call_target, [matcher(node.call_target)
        #                          for matcher in method_matchers],
        #       file=sys.stderr)
        if not any(matcher(node.call_target) for matcher in method_matchers):
            try_remove_node(graph, node)
            continue


def get_full_graph(args, graphs):
    logging.debug("getting graph for method %r", args.method)

    if args.type_overrides:
        logging.debug("loading type overrides")
        with args.type_overrides as f:
            overrides = load_type_overrides(f)

        logging.debug("using type overrides: %r", overrides)

        graphs.add_filter(functools.partial(
            apply_type_overrides,
            overrides,
        ))

    logging.debug("loading graph")
    cdfg = graphs[args.method]

    if args.inline:
        logging.debug("inlining method calls as far as possible")
        inline_calls(args.method, cdfg, graphs)

    if args.simplify:
        logging.debug("simplifying graph")
        cdfg.strip_exceptional()
        cdfg.simplify_basic_blocks()
        strip_asm_stack_instructions(cdfg)

    if args.strip_non_calls:
        logging.debug("loading list of method calls to keep")
        with args.strip_non_calls as f:
            method_matchers = load_method_matchers(f)

        logging.debug("using method matchers: %r", method_matchers)

        strip_non_calls(method_matchers, cdfg)

    logging.debug("graph for %r loaded", args.method)
    return cdfg


def cdfg_dot(args,
             graphs: typing.Mapping[
                 str,
                 prettycdfg.nodes.ControlDataFlowGraph]):
    cdfg = get_full_graph(args, graphs)

    if args.outfile is None:
        f = sys.stdout
    else:
        f = open(args.outfile, "w")

    def emit_inputs(node, f):
        for i, input_ in enumerate(node.inputs):
            print(
                '"{}" -> "{}" [style=dashed,headlabel="in {}",color=blue]'.format(
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
                    str(floating),
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
                        str(node),
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


def cdfg_xml(args,
             graphs: typing.Mapping[
                 str,
                 prettycdfg.nodes.ControlDataFlowGraph]):
    cdfg = get_full_graph(args, graphs)

    tree = prettycdfg.nodes.save_asm(cdfg)
    tree.set("id", args.method)

    if args.outfile is None:
        f = sys.stdout.buffer.raw
    else:
        f = open(args.outfile, "wb")

    with f:
        tree.getroottree().write(f)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Global options")
    group.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        default=0,
        help="Increase verbosity (up to -vvv)"
    )

    group = parser.add_argument_group("Input processing options")
    group.add_argument(
        "--inline",
        action="store_true",
        default=False,
        help="Inline all the function calls",
    )
    group.add_argument(
        "--no-simplify",
        default=True,
        dest="simplify",
        action="store_false",
    )
    group.add_argument(
        "--type-overrides",
        default=None,
        type=argparse.FileType("r"),
        help="File with field type overrides to apply to invocations",
        metavar="FILE",
    )
    group.add_argument(
        "--strip-non-calls",
        default=None,
        type=argparse.FileType("r"),
        help="Strip all non-floating nodes which are not function calls to "
        "the given classes, interfaces or methods.",
        metavar="FILE",
    )

    parser.add_argument(
        "graph",
        type=argparse.FileType("rb"),
        help="XML CDFG input"
    )

    subparsers = parser.add_subparsers(title="Command")

    subparser = subparsers.add_parser(
        "bb-graph-dot",
        description="Plot dot graph of the basic blocks in the selected method."
    )
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

    subparser = subparsers.add_parser(
        "cdfg-dot",
        description="Print the control- and data-flow graph of the method as "
        "dot."
    )
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

    subparser = subparsers.add_parser(
        "cdfg-xml",
        description="Print an XML file which contains only the CDFG of the "
        "method (this is useful when input processing options are used)."
    )
    subparser.set_defaults(func=cdfg_xml)
    subparser.add_argument(
        "method",
        help="Method to dump the graph for"
    )
    subparser.add_argument(
        "outfile",
        nargs="?",
        help="File to dump the XML graph to"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level={
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
        }.get(args.verbosity, logging.DEBUG),
    )

    if not hasattr(args, "func"):
        print("a subcommand is required", file=sys.stderr)
        parser.print_help()
        return 1

    logging.debug("opening graph data from %s", args.graph.name)
    methods = load_graphs(args.graph)
    args.func(args, methods)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
