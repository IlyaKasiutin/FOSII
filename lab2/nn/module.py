class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
