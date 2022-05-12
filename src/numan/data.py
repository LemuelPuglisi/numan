
class Point:
    def __init__(self, node, value, grad=None):
        self.node   = node
        self.value  = value
        self.grad   = grad

    def set_grad(self, grad):
        self.grad = grad