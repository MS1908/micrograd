import numpy as np


def unbroadcast_grad(grad, target_shape):
    orig_shape = grad.shape
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
        
    for i in range(grad.ndim):
        if target_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    if grad.shape != target_shape:
        raise ValueError(f"Cannot unbroadcast grad with shape {orig_shape} to {target_shape}")
    
    return grad


class Value:

    def __init__(self, data, _children=()):
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Can not convert data type {type(data)} to numpy array")
            
        if data.ndim > 2:
            raise ValueError(f"Data of value must have number of dimensions less than 3, got {data.ndim} dimensions")
        
        self.data = data.astype(np.float64)
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        
    def __repr__(self):
        return f"data={self.data}\ngrad={self.grad}"
    
    def __str__(self):
        return f"data={self.data}\ngrad={self.grad}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other))
        
        def _backward():
            self.grad += unbroadcast_grad(out.grad, self.data.shape)
            other.grad += unbroadcast_grad(out.grad, other.data.shape)
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        out = Value(out_data, (self, other))

        def _backward():
            self.grad += unbroadcast_grad(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast_grad(self.data * out.grad, other.data.shape)
        
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers"
        out_data = self.data ** other
        out = Value(out_data, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data @ other.data
        out = Value(out_data, (self, other))
        
        def _backward():
            if self.data.ndim == 2 and other.data.ndim == 2:
                # A (m,n) @ B (n,p) -> out (m,p) => grad_a shape (m,n), grad_b shape (n,p)
                # grad (m,p) @ B.T (p,n) -> grad_a (m,n)
                # A.T (n,m) @ grad (m,p) -> grad_b (n,p)
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            elif self.data.ndim == 1 and other.data.ndim == 2:
                # A (m,) @ B (m,n) -> out (n,) => grad_a shape (m,), grad_b shape (m,n)
                # grad (n,) @ B.T (n,m) -> grad_a (m,) -- OK
                # A.T (m,) @ grad (n,) -> ? (Because of transpose of vector in numpy leave the vector unchange)
                # So we use outer product for A.T and grad here
                self.grad += out.grad @ other.data.T
                other.grad += np.outer(self.data, out.grad)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                # A (m,n) @ B (n,) -> out (m,) => grad_a shape (m,n), grad_b shape (n,)
                # A.T (n,m) @ grad (m,) -> grad_b (n,) -- OK
                # grad (m,) @ B.T (n,) -> ?
                # Similarly, we use outer product
                self.grad += np.outer(out.grad, other.data)
                other.grad += self.data.T @ out.grad
            else:  # Only case remain: Both data have dim 1 -> Dot product
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
                
        out._backward = _backward
        return out
