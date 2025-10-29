class Module:
    def __init__(self):
        self._params = {}
        self._modules = {}

    def parameters(self):
        for p in self._params.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        # For layers with gradient attributes, zero them out
        if hasattr(self, 'W_grad'):
            self.W_grad.fill(0)
        if hasattr(self, 'bias_grad'):
            self.bias_grad.fill(0)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
