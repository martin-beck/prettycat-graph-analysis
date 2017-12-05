Graph Analysis Tool for Java ASM-based Data and Control Flow graphs
###################################################################

TL;DR: This tool takes XML-based graphs representing control and data flow of compiled Java programs an does things (dotting, simplification, inlining, …) with them.

The Context
===========

This tool is part of the prettycat suite. It processes output from the `ASM Dataflow`_ tool. The ASM Dataflow tool operates on compiled java programs (provided either as .class, .jar or mix of both) and extracts a control and data flow graph represented as XML.

Graph Analysis Tool
===================

The input XML contains a representation of methods on the bytecode level. Each instruction has a set of data flow inputs and a set of control flow exits.

The graph analysis tool always operates on a target method from the input XML. Each method which is loaded (more than one may be loaded from the XML when inlining is used) is put into an in-memory representation of the graph. Right after loading, the basic blocks of the method are inferred.

On this graph, a few different processing operations can be carried out (in
this order):

* ``--type-overrides`` takes a file which maps type names to other type names.
  All calls which call methods of a type occuring on the left hand of the
  mapping are re-written to the corresponding right hand side. This can be used
  to force specific implementations of interfaces to be assumed.

* ``--inline`` will inline all method calls for which a graph is found in the
  input XML. If methods recurse to themselves, inlining is aborted for that
  subtree (other methods are still inlined).

* (by default, disable with ``--no-simplify``) Exceptional control flow edges
  are removed and basic blocks are re-simplified. All ASM instructions which
  only modify the stack are stripped (the data flow caused by these
  instructions has already been analyzed by the ASM tool).

* ``--strip-non-calls`` takes a file with a list of globs. All non-call
  instructions and all calls to methods which do **not** match those globs are
  removed from the graph. After this operation, the data flow is meaningless.
  The graph is then only useful for very coarse control flow analysis.

The following generation operations exist (only one can be chosen per
invocation):

* ``cdfg-xml``: Generate an XML file with the resulting graph which can be used
  as input to the analysis tool again. This is useful if different graphing
  operations are to be tried.

* ``cdfg-dot``: Plot the whole CDFG as dot file.

* ``bb-graph-dot``: Plot the basic blocks of the CDFG and their control flow
  as dot file.
