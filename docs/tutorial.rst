Tutorial
########

This is a walk-through for analysing Java OTR with PrettyCat.

Prerequisites
=============

* You need Java 8 and Eclipse installed. The specific version of eclipse does
  not matter.
* Graphviz
* General knowledge with shell and git.


Obtaining a built version of Java OTR
=====================================

You can either build Java OTR by yourself, or download a ready-made tarball
including .class files::

    wget https://sotecware.net/files/noindex/java-otr-built.tar.gz

Extract the tarball in a directory, so that ``…/java-otr/classes`` is a valid
path (we will refer to that path later).


Setting up and running dataflow-asm
===================================

First, you need to clone the dataflow-asm repository::

    git clone https://github.com/martin-beck/dataflow-asm

Then you need to import this project into eclipse (
File -> Import, General -> Import existing Projetc into workspace). The project
should be buildable immediately.

You need to add a Run confuguration:

1. Run -> Run Configurations
2. Double-click ``Java Application``
3. Select the ``dataflow-asm`` project (via the ``Browse`` button next to the
   Project input)
4. Select the ``org.prettycat.dataflow.asm.DataflowAnalyser`` main class (via
   the ``Search`` button next to the Main class input)
5. In the Arguments tab, enter the following program arguments::

        -o out.xml -p …/java-otr/classes ca.uwaterloo.crysp.otr.demo.Driver ca.uwaterloo.crysp.otr.demo.SendingThread ca.uwaterloo.crysp.otr.demo.ReceivingThread ca.uwaterloo.crysp.otr.UserState ca.uwaterloo.crysp.otr.ConnContext

    (remember what we said earlier about the java-otr path)

    (``-o``: Output file; ``-p``: Path to .class files/.jar file; everything
     else: classes to analyze)

6. Click ``Apply``.
7. Click ``Run``.

You should see a bunch of output, some of which will be red and stack-tracy.
That is fine, as long as you end up with a few megabytes (somewhere around 8) of
``out.xml`` in the ``dataflow-asm`` directory.


Setting up and running prettycat-ga
===================================

Clone the prettycat-graph-analysis repository::

    git clone https://github.com/martin-beck/prettycat-graph-analysis

We will assume that you copied the ``out.xml`` from the previous section into
the ``prettycat-graph-analysis`` directory, but that is not mandatory; whereever
we refer to ``out.xml``, you can also simply put the full/relative path to the
file.

The general syntax of the ``prettycat-ga.py`` command is::

    python3 prettycat-ga.py [global-options] [infile-options] infile command [command-options]

While ``global-options`` and ``infile-options`` can be mixed, they can not
occur after the ``command``. Likewise, ``command-options`` cannot occur before
the command.

Let’s have a bit of fun with those commands. First::

    python3 prettycat-ga.py out.xml bb-graph-dot 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' | dot -Tsvg -o /tmp/foo.svg

(this will take a moment, ``dot`` isn’t the fastest) and open ``/tmp/foo.svg``
in your favourite SVG viewer (in a pinch, your web browser will do the job,
too).

This shows the simplified basic blocks of the method, along with the data flows
(blue) between them. The basic block names are based on the instruction at
which they start, aside from that there’s not much useful.

In addition to the basic blocks, there are also so called "floating nodes".
Floating nodes are things which are not attached to any basic block. Those are
data-flow only primitives and can represent exceptions, parameters, as well as
merge nodes (merge nodes join multiple data flows from different basic blocks
entering a single basic block).

This graph may or may not be useful. To look at how a method looks in detail,
a full cdfg is needed::

    python3 prettycat-ga.py out.xml cdfg-dot -g 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' | dot -Tsvg -o /tmp/foo.svg

This plot contains all instructions and floating nodes of a method, along with
the data- (blue, dashed) and control- (black) flow between them. Some
instructions which are irrelevant to any analysis or which have been analysed
by ASM already are stripped by default (to disable this feature, use
``--no-simplify`` as input option; be aware though that this also disables
stripping of exception control flows, which can get nasty quickly).

In addition, the nodes are grouped into blocks according to their basic block
membership (caused by the ``-g`` option). Note that the grouping is only
approximate -- floating nodes may penetrate into basic blocks if the layout
algorithm decides so.

Now you’ll notice that for in-depth analysis, it would be useful to have method
calls inlined. Fear not:

    python3 prettycat-ga.py --inline out.xml cdfg-dot -g 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' | dot -Tsvg -o /tmp/foo.svg

``--inline`` causes the tool to inline any method call for which it can find a
CDFG in the input XML file into the graph (note: self-recursion is not inlined).
You expect a monster of a graph after this command, but unfortunately, this is
not the case.

The reason is that Java OTR makes use of interfaces a lot. However, the rather
primitive data flow analysis used by this tools cannot resolve the interface
type to the actual type automatically. This is where type overrides come into
play: this is essentialy a list of type->type mappings which are applied to all
calls in the graph. It allows the analysis to assume which implementation of
an interface is used, and thus perform more inlining than without that
information.

To make use of this, create a file ``otr.typeoverrides`` with the following
contents::

    java:ca.uwaterloo.crysp.otr.iface.OTRInterface   java:ca.uwaterloo.crysp.otr.UserState
    java:ca.uwaterloo.crysp.otr.iface.OTRCallbacks   java:ca.uwaterloo.crysp.otr.demo.LocalCallback
    java:ca.uwaterloo.crysp.otr.iface.OTRContext     java:ca.uwaterloo.crysp.otr.ConnContext

The left hand side of the "table" is the original type the right hand side is
the type which will be used instead.

To make use of this file, invoke the tool again. But this time, we will use the
``cdfg-xml`` instead of the ``cdfg-dot`` command. The former will create an XML
file which can be fed to the tool again. We do this, because the resulting
graph is way too large be useful. We will instead process it further.

    python3 prettycat-ga.py -vvv --inline --type-overrides otr.typeoverrides out.xml cdfg-xml 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' > inlined.xml

(We invoke with ``-vvv`` to get some sense of progress.)

You will notice that the runtime increases quite a bit. This is simply due to
the high amount of work which can now be done; a lot of methods can now be
inlined, thus the graph grows a lot more.

The log output will show where type overrides are applied, and also where
inlining fails to do its work due to lack of graphs or recursion. A lack of
graphs is normal, because we did not (for example) analyse the Java standard
library. In production use, you’ll want to grep for any function which you
expected to be inlined.

As mentioned, the resulting graph is not really useful. It contains a few ten
thousand of nodes and basic blocks. To reduce this further, we’ll want to focus
on methods which are of interest for us. For this, we create a file with
shell-style globs which tell the tool which methods are of interest. We call
that file ``otr.keep``, and for now it has only a single line::

    java:ca.uwaterloo.crysp.otr.crypt.*

We continue to operate on the graph we just created::

    python3 prettycat-ga.py -vv --strip-non-calls otr.keep inlined.xml cdfg-xml 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' > stripped.xml

This should be a bit faster than the previous command, and the resulting graph
should be much smaller. Note that data flow information gets lost in this step.
The resulting graph contains only a partial data flow, which must not be used
for critical analysis. It is useful for visualisation though::

    python3 prettycat-ga.py stripped.xml cdfg-dot 'java:ca.uwaterloo.crysp.otr.demo.SendingThread.run[()V]' | dot -Tsvg -o /tmp/foo.svg
