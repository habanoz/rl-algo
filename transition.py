class Transition:
    def __init__(self, s, a, r, ns, flags=None):
        self.s = s
        self.a = a
        self.r = r
        self.ns = ns
        self.flags = flags

    def to_tuple(self):
        return self.s, self.a, self.r, self.ns, self.flags
