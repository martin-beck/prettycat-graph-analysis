TARGETS    = DH.bgv

JAVA_OPTS += -Dgraal.Dump=
JAVA_OPTS += -Dgraal.PrintCFG=true
JAVA_OPTS += -XX:+UseJVMCICompiler
JAVA_OPTS += -XX:+PrintCompilation
JAVA_OPTS += -Xcomp
JAVA_OPTS += -XX:-TieredCompilation
JAVA_OPTS += -Dgraal.PrintGraphFile=true

# Limit compilation to class/method (probably not useful)
#JAVA_OPTS += -XX:CompileOnly=DH

all: $(TARGETS)

clean:
	rm -f $(TARGETS) *.bgv *.cfg DH.log *.class

%.bgv: %.class
	$(shell which java) -Dgraal.PrintGraphFileName=$* $(JAVA_OPTS) $*

%.class: %.java
	$(shell which javac) $(JAVAC_OPTS) $<

%.jar: %.class
	jar cf $@ $<
