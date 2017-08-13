import contextlib
import unittest
import unittest.mock

import prettycdfg.nodes as nodes
import prettycdfg.xmlutil as xmlutil

import lxml.builder


class TestNode(unittest.TestCase):
    def setUp(self):
        self.uid = "foo"
        self.n = nodes.Node(self.uid)

    def tearDown(self):
        del self.n

    def test_init_default(self):
        with self.assertRaisesRegex(TypeError, "unique_id"):
            nodes.Node()

    def test_init(self):
        self.assertEqual(self.n.unique_id, self.uid)
        self.assertCountEqual(self.n.inputs, [])

    def test_unique_id_is_read_only(self):
        with self.assertRaisesRegex(AttributeError, "can't set"):
            self.n.unique_id = self.n.unique_id

    def test_predecessors(self):
        self.assertCountEqual(self.n.predecessors, [])

    def test_predecessors_is_read_only(self):
        with self.assertRaisesRegex(AttributeError, "can't set"):
            self.n.predecessors = []

    def test_successors(self):
        self.assertCountEqual(self.n.successors, [])

    def test_successors_is_read_only(self):
        with self.assertRaisesRegex(AttributeError, "can't set"):
            self.n.successors = []

    def test_block(self):
        self.assertIsNone(self.n.block)

    def test_block_is_read_only(self):
        with self.assertRaisesRegex(AttributeError, "can't set"):
            self.n.block = self.n.block


class TestBasicBlock(unittest.TestCase):
    def setUp(self):
        self.uid = "bar"
        self.bb = nodes.BasicBlock(self.uid)

    def tearDown(self):
        del self.bb

    def test_init_default(self):
        with self.assertRaisesRegex(TypeError, "unique_id"):
            nodes.BasicBlock()

    def test_init(self):
        self.assertEqual(self.bb.unique_id, self.uid)
        self.assertCountEqual(self.bb.nodes, [])
        self.assertCountEqual(self.bb.exits, [])

    def test_unique_id_is_read_only(self):
        with self.assertRaisesRegex(AttributeError, "can't set"):
            self.bb.unique_id = self.bb.unique_id


class TestControlDataFlowGraph(unittest.TestCase):
    def setUp(self):
        self.cdfg = nodes.ControlDataFlowGraph()

    def tearDown(self):
        del self.cdfg

    def test_new_block(self):
        self.assertCountEqual(self.cdfg.blocks, [])

        with contextlib.ExitStack() as stack:
            uuid4 = stack.enter_context(unittest.mock.patch("uuid.uuid4"))
            uuid4.return_value = 2

            block = self.cdfg.new_block()

        uuid4.assert_called_once_with()

        self.assertEqual(block.unique_id, "urn:uuid:2")
        self.assertCountEqual(self.cdfg.blocks, [block])

    def test_new_node_without_block(self):
        self.assertCountEqual(self.cdfg.nodes, [])

        with contextlib.ExitStack() as stack:
            constructor = unittest.mock.Mock()

            uuid4 = stack.enter_context(unittest.mock.patch("uuid.uuid4"))
            uuid4.return_value = 3

            node = self.cdfg.new_node(constructor)

        uuid4.assert_called_once_with()

        constructor.assert_called_once_with("urn:uuid:3")

        self.assertEqual(node, constructor())
        self.assertCountEqual(self.cdfg.nodes, [node])
        self.assertCountEqual(self.cdfg.floating_nodes, [node])

    def test_new_node_with_block(self):
        bb = self.cdfg.new_block()

        self.assertCountEqual(self.cdfg.nodes, [])

        with contextlib.ExitStack() as stack:
            constructor = unittest.mock.Mock()

            uuid4 = stack.enter_context(unittest.mock.patch("uuid.uuid4"))
            uuid4.return_value = 3

            node = self.cdfg.new_node(constructor, block=bb)

        uuid4.assert_called_once_with()

        constructor.assert_called_once_with("urn:uuid:3")

        self.assertEqual(node, constructor())
        self.assertCountEqual(self.cdfg.nodes, [node])
        self.assertCountEqual(bb.nodes, [node])
        self.assertCountEqual(self.cdfg.floating_nodes, [])

    def test_block_by_node(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertIs(n1.block, bb1)
        self.assertIs(n2.block, bb2)
        self.assertIs(bb1, self.cdfg.block_by_node(n1))
        self.assertIs(bb2, self.cdfg.block_by_node(n2))

    def test_add_second_node(self):
        bb1 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb1)

        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n1.successors, [n2])
        self.assertCountEqual(n2.predecessors, [n1])
        self.assertCountEqual(n2.successors, [])

    def test_block_by_node_with_floating_node(self):
        n = self.cdfg.new_node(nodes.Node)

        self.assertIsNone(n.block)

        with self.assertRaises(KeyError):
            self.cdfg.block_by_node(n)

    def test_add_successor(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertCountEqual(bb1.exits, [])
        self.assertCountEqual(n1.successors, [])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [])

        self.cdfg.add_successor(n1, n2)

        self.assertCountEqual(bb1.exits, [bb2])
        self.assertCountEqual(n1.successors, [n2])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [n1])

    def test_add_successor_at_floating(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertCountEqual(bb1.exits, [])

        with self.assertRaisesRegex(ValueError,
                                    "cannot add successor to floating node"):
            self.cdfg.add_successor(n1, n2)

        self.assertCountEqual(n1.successors, [])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [])

    def test_add_floating_successor(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node)

        self.assertCountEqual(bb1.exits, [])

        with self.assertRaisesRegex(ValueError,
                                    "cannot add floating node as successor"):
            self.cdfg.add_successor(n1, n2)

        self.assertCountEqual(bb1.exits, [])

        self.assertCountEqual(n1.successors, [])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [])

    def test_add_successor_at_non_last(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n11 = self.cdfg.new_node(nodes.Node, block=bb1)

        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertCountEqual(bb1.exits, [])

        with self.assertRaisesRegex(ValueError,
                                    "cannot add successor to non-last node"):
            self.cdfg.add_successor(n1, n2)

        self.assertCountEqual(bb1.exits, [])

        self.assertCountEqual(n1.successors, [n11])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [])

    def test_add_successor_to_non_first(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)

        n20 = self.cdfg.new_node(nodes.Node, block=bb2)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertCountEqual(bb1.exits, [])

        with self.assertRaisesRegex(ValueError,
                                    "cannot add non-first node as successor"):
            self.cdfg.add_successor(n1, n2)

        self.assertCountEqual(bb1.exits, [])

        self.assertCountEqual(n1.successors, [])
        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n2.successors, [])
        self.assertCountEqual(n2.predecessors, [n20])

    def test_split_block(self):
        bb1 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb1)
        n3 = self.cdfg.new_node(nodes.Node, block=bb1)

        bb2 = self.cdfg.split_block(n2)

        self.assertIsNot(n2.block, bb1)
        self.assertIsNot(n3.block, bb1)

        self.assertIs(n2.block, bb2)
        self.assertIs(n2.block, n3.block)

        self.assertCountEqual(bb1.exits, [bb2])
        self.assertCountEqual(n1.successors, [n2])
        self.assertCountEqual(n2.predecessors, [n1])
        self.assertSequenceEqual(list(bb1.nodes), [n1])
        self.assertSequenceEqual(list(bb2.nodes), [n2, n3])

        self.assertTrue(bb2.unique_id)

    def test_split_block_with_exit(self):
        bb1 = self.cdfg.new_block()
        bb3 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb1)
        n3 = self.cdfg.new_node(nodes.Node, block=bb3)

        self.cdfg.add_successor(n2, n3)

        bb2 = self.cdfg.split_block(n2)

        self.assertIs(n1.block, bb1)
        self.assertIs(n2.block, bb2)
        self.assertIs(n3.block, bb3)

        self.assertIsNot(n1.block, bb2)
        self.assertIsNot(n1.block, bb3)

        self.assertIsNot(n2.block, bb1)
        self.assertIsNot(n2.block, bb3)

        self.assertIsNot(n3.block, bb1)
        self.assertIsNot(n3.block, bb2)

        self.assertNotEqual(bb2.unique_id, bb3.unique_id)
        self.assertNotEqual(bb2.unique_id, bb1.unique_id)

        self.assertCountEqual(bb1.exits, [bb2])
        self.assertCountEqual(bb2.exits, [bb3])

        self.assertCountEqual(n1.successors, [n2])
        self.assertCountEqual(n2.successors, [n3])

        self.assertCountEqual(n2.predecessors, [n1])
        self.assertCountEqual(n3.predecessors, [n2])

    def test_add_input_both_floating(self):
        n1 = self.cdfg.new_node(nodes.Node)
        n2 = self.cdfg.new_node(nodes.Node)

        self.assertCountEqual(n1.inputs, [])

        self.cdfg.add_input(n1, n2)

        self.assertCountEqual(n1.inputs, [n2])

    def test_node_by_id(self):
        bb1 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node)

        self.assertIs(self.cdfg.node_by_id(n1.unique_id), n1)
        self.assertIs(self.cdfg.node_by_id(n2.unique_id), n2)

    def test_node_by_id_raises_KeyError_on_unknown_id(self):
        bb1 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node)

        with self.assertRaises(KeyError):
            self.cdfg.node_by_id("foo")

    def test_join_blocks_b2_empty(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)

        self.assertIs(self.cdfg.join_blocks(bb1, bb2), bb1)

        self.assertCountEqual(self.cdfg.blocks, [bb1])
        self.assertCountEqual(self.cdfg.nodes, [n1])
        self.assertCountEqual(bb1.nodes, [n1])
        self.assertCountEqual(bb2.nodes, [])
        self.assertIs(self.cdfg.block_by_node(n1), bb1)

    def test_join_blocks_b1_empty(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.assertIs(self.cdfg.join_blocks(bb1, bb2), bb1)

        self.assertCountEqual(self.cdfg.blocks, [bb1])
        self.assertCountEqual(self.cdfg.nodes, [n1])
        self.assertCountEqual(bb1.nodes, [n1])
        self.assertCountEqual(bb2.nodes, [])
        self.assertIs(self.cdfg.block_by_node(n1), bb1)

    def test_join_blocks_nonempty_consecutive(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)

        self.cdfg.add_successor(n1, n2)

        self.assertIs(self.cdfg.join_blocks(bb1, bb2), bb1)

        self.assertCountEqual(self.cdfg.blocks, [bb1])
        self.assertCountEqual(self.cdfg.nodes, [n1, n2])
        self.assertCountEqual(bb1.nodes, [n1, n2])
        self.assertCountEqual(bb2.nodes, [])
        self.assertIs(self.cdfg.block_by_node(n1), bb1)
        self.assertIs(self.cdfg.block_by_node(n2), bb1)
        self.assertCountEqual(n1.successors, [n2])
        self.assertCountEqual(n2.predecessors, [n1])

    def test_join_blocks_incorrect_successor(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()
        bb3 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)
        n3 = self.cdfg.new_node(nodes.Node, block=bb3)

        self.cdfg.add_successor(n1, n2)
        self.cdfg.add_successor(n2, n3)

        with self.assertRaisesRegex(
                ValueError,
                r"b2 is not a successor of b1"):
            self.cdfg.join_blocks(bb1, bb3)

    def test_join_blocks_multi_successors(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()
        bb3 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)
        n3 = self.cdfg.new_node(nodes.Node, block=bb3)

        self.cdfg.add_successor(n1, n2)
        self.cdfg.add_successor(n1, n3)

        with self.assertRaisesRegex(
                ValueError,
                r"b1 has multiple successors"):
            self.cdfg.join_blocks(bb1, bb2)

    def test_join_blocks_multi_predecessors(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()
        bb3 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)
        n3 = self.cdfg.new_node(nodes.Node, block=bb3)

        self.cdfg.add_successor(n1, n2)
        self.cdfg.add_successor(n3, n2)

        with self.assertRaisesRegex(
                ValueError,
                r"b2 has multiple predecessors"):
            self.cdfg.join_blocks(bb1, bb2)

    def test_simplify_basic_blocks(self):
        bb1 = self.cdfg.new_block()
        bb2 = self.cdfg.new_block()
        bb3 = self.cdfg.new_block()
        bb4 = self.cdfg.new_block()
        bb5 = self.cdfg.new_block()

        n1 = self.cdfg.new_node(nodes.Node, block=bb1)
        n2 = self.cdfg.new_node(nodes.Node, block=bb2)
        n3 = self.cdfg.new_node(nodes.Node, block=bb3)
        n4 = self.cdfg.new_node(nodes.Node, block=bb4)
        n5 = self.cdfg.new_node(nodes.Node, block=bb5)

        self.cdfg.add_successor(n1, n2)
        self.cdfg.add_successor(n1, n3)
        self.cdfg.add_successor(n2, n4)
        self.cdfg.add_successor(n3, n4)
        self.cdfg.add_successor(n4, n5)

        self.cdfg.simplify_basic_blocks()

        self.assertCountEqual(self.cdfg.blocks, [bb1, bb2, bb3, bb4])
        self.assertCountEqual(self.cdfg.nodes, [n1, n2, n3, n4, n5])
        self.assertCountEqual(bb1.nodes, [n1])
        self.assertCountEqual(bb2.nodes, [n2])
        self.assertCountEqual(bb3.nodes, [n3])
        self.assertCountEqual(bb4.nodes, [n4, n5])
        self.assertCountEqual(bb5.nodes, [])
        self.assertIs(self.cdfg.block_by_node(n4), bb4)
        self.assertIs(self.cdfg.block_by_node(n5), bb4)


class Testload_asm(unittest.TestCase):
    def setUp(self):
        self.E = lxml.builder.ElementMaker()

    def tearDown(self):
        del self.E

    def test_load_empty(self):
        tree = self.E(xmlutil.ASM.method)

        result = nodes.load_asm(tree)

        self.assertCountEqual(result.blocks, [])
        self.assertCountEqual(result.nodes, [])

    def test_load_single_instruction(self):
        tree = self.E(
            xmlutil.ASM.method,
            self.E(
                xmlutil.ASM.insns,
                self.E(
                    xmlutil.ASM.insn,
                    id="java:test",
                    opcode="123",
                    line="2",
                )
            )
        )

        result = nodes.load_asm(tree)

        bb1, = result.blocks
        node, = result.nodes

        self.assertCountEqual(bb1.nodes, [node])

        self.assertEqual(node.line, 2)
        self.assertEqual(node.opcode, 123)

    def test_single_bb_instructions(self):
        tree = self.E(
            xmlutil.ASM.method,
            self.E(
                xmlutil.ASM.insns,
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/3",
                        )
                    ),
                    id="java:test/1",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    id="java:test/2",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/2",
                        )
                    ),
                    id="java:test/3",
                )
            )
        )

        result = nodes.load_asm(tree)

        bb1, = result.blocks
        n1, n2, n3, = result.nodes

        self.assertCountEqual(bb1.nodes, [n1, n2, n3])

    def test_multi_bb(self):
        tree = self.E(
            xmlutil.ASM.method,
            self.E(
                xmlutil.ASM.insns,
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/2",
                        ),
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/3",
                        )
                    ),
                    id="java:test/1",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/4",
                        )
                    ),
                    id="java:test/2",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/4",
                        )
                    ),
                    id="java:test/3",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.exits,
                        self.E(
                            xmlutil.ASM.exit,
                            to="java:test/5",
                        )
                    ),
                    id="java:test/4",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    id="java:test/5",
                ),
            )
        )

        result = nodes.load_asm(tree)

        n1 = result.node_by_id("java:test/1")
        n2 = result.node_by_id("java:test/2")
        n3 = result.node_by_id("java:test/3")
        n4 = result.node_by_id("java:test/4")
        n5 = result.node_by_id("java:test/5")

        bb1 = result.block_by_node(n1)
        bb2 = result.block_by_node(n2)
        bb3 = result.block_by_node(n3)
        bb4 = result.block_by_node(n4)

        self.assertCountEqual(n1.predecessors, [])
        self.assertCountEqual(n1.successors, [n2, n3])

        self.assertCountEqual(n2.predecessors, [n1])
        self.assertCountEqual(n2.successors, [n4])

        self.assertCountEqual(n3.predecessors, [n1])
        self.assertCountEqual(n3.successors, [n4])

        self.assertCountEqual(n4.predecessors, [n2, n3])
        self.assertCountEqual(n4.successors, [n5])

        self.assertCountEqual(n5.predecessors, [n4])
        self.assertCountEqual(n5.successors, [])

        self.assertCountEqual(bb1.nodes, [n1])
        self.assertCountEqual(bb2.nodes, [n2])
        self.assertCountEqual(bb3.nodes, [n3])
        self.assertCountEqual(bb4.nodes, [n4, n5])

    def test_inputs(self):
        tree = self.E(
            xmlutil.ASM.method,
            self.E(
                xmlutil.ASM.insns,
                self.E(
                    xmlutil.ASM.insn,
                    id="java:test/1",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    id="java:test/2",
                ),
                self.E(
                    xmlutil.ASM.insn,
                    self.E(
                        xmlutil.ASM.inputs,
                        self.E(
                            xmlutil.ASM("value-of"),
                            **{"from": "java:test/1"},
                        ),
                        self.E(
                            xmlutil.ASM("value-of"),
                            **{"from": "java:test/2"},
                        ),
                    ),
                    id="java:test/3",
                )
            )
        )

        result = nodes.load_asm(tree)

        n1, n2, n3 = result.nodes

        self.assertCountEqual(n3.inputs, [n1, n2])

    def test_parameter_nodes(self):
        tree = self.E(
            xmlutil.ASM.method,
            self.E(
                xmlutil.ASM.parameters,
                self.E(
                    xmlutil.ASM.parameter,
                    id="java:param/0",
                    type_="java:I",
                ),
                self.E(
                    xmlutil.ASM.parameter,
                    id="java:param/1",
                    type_="java:Lorg/foo;"
                ),
            )
        )

        result = nodes.load_asm(tree)

        n1, n2 = result.nodes

        self.assertIsInstance(n1, nodes.ParameterNode)
        self.assertEqual(n1.type_, "java:I")
        self.assertIsNone(n1.block)

        self.assertIsInstance(n2, nodes.ParameterNode)
        self.assertEqual(n2.type_, "java:Lorg/foo;")
        self.assertIsNone(n2.block)
