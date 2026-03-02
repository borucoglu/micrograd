import math as m
from random import uniform

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def show(self):
        print(self.__repr__())
    
    def __add__(val1,val2):
        out = Value(val1.data + val2.data, (val1,val2),'+')

        def _backward():
            val1.grad += out.grad 
            val2.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(val1,val2):
        out = Value(val1.data * val2.data, (val1,val2),'*')

        def _backward():
            val1.grad += val2.data* out.grad
            val2.grad += val1.data* out.grad
        
        out._backward = _backward 
        return out
    
    def __sub__(val1,val2):
        out = Value(val1.data - val2.data, (val1,val2),'-')

        def _backward():
            val1.grad += out.grad 
            val2.grad -= out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(val1,val2):
        out = Value(val1.data / val2.data, (val1,val2),'/')

        def _backward():
            val1.grad += 1/val2.data * out.grad
            val2.grad += -val1.data / (val2.data**2) * out.grad 

        out._backward = _backward
        return out  
    
    def exp(self):
        out = Value(m.exp(self.data), (self,),'e')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        if self.data > 0:
            val = self.data
        else:
            val = 0
        out = Value(val,(self,),'relu')

        def _backward():
            if self.data > 0:
                dself = 1
            else: dself = 0
            self.grad += dself * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(m.tanh(self.data),(self,),'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out 
        
    
    def backward(self):
        
        visited = set()
        top_order = []

        def back_rec(self):
                if self not in visited:
                    visited.add(self)
                    for child in self._prev:
                        back_rec(child)
                    top_order.append(self)
        self.grad = 1.0
        back_rec(self)

        for node in reversed(top_order):
            node._backward()
        print(self.grad)

class Neuron:
    def __init__(self,nb_inputs,activation = Value.relu):
        self.nb_in = nb_inputs
        self.w = [Value(uniform[-1,1]) for _ in range(nb_inputs)]
        self.b = Value(uniform[-1,1])
        self.activation = activation

    def compute_sum(self, inputs):
        sum = 0
        for i in range(self.nb_in):
            sum += inputs[i] * self.w[i]
        return (sum + self.b).activation()
    
    def parameters(self):
        return (self.w,self.b)


