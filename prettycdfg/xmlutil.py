class Namespace:
    def __init__(self, ns):
        super().__init__()
        self.__dict__["namespace"] = ns

    @property
    def xmlns(self):
        return str(self)

    def __str__(self):
        return self.__dict__["namespace"]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return self(name)

    def __call__(self, name):
        return "{{{}}}{}".format(self.__dict__["namespace"], name)


IrGraph = Namespace("https://xmlns.zombofant.net/prettycat/1.0/ir-graph")
ASM = Namespace("https://xmlns.zombofant.net/prettycat/1.0/asm")
