import numpy as np
import math


def unbroadcast_grad(target_shape, grad):  # In unbroadcast, grad.ndim >= target.ndim
    if grad.ndim < len(target_shape):
        raise ValueError(f"Target ndim {len(target_shape)} should never exceed grad.ndim {grad.ndim} when unbroadcast.")
    
    if grad.shape == target_shape:
        return grad
    
    if target_shape == ():
        # Broadcasting scalar to vector/matrices -> Unbroadcast back to scalar
        return np.sum(grad)
    
    if len(target_shape) == 1:
        if grad.ndim == 1:  # grad shape != target shape
            if target_shape[0] == 1 and grad.shape[0] > 1:
                # Broadcasting (1,) to (N,) -> Unbroadcast (N,) to (1,). 
                # Basically broadcast scalar to vector
                return np.sum(grad, keepdims=True)
            else:
                raise ValueError(f"Cannot unbroadcast 1D grad {grad.shape} to 1D target {target_shape}")

        elif grad.ndim == 2:
            # Broadcast (R, C) back to (R, 1) or (1, C)
            if target_shape[0] == grad.shape[1]:
                return grad.sum(axis=0)
            elif target_shape[0] == grad.shape[0]:
                return grad.sum(axis=1)
            else:
                raise ValueError(f"Cannot unbroadcast 2D grad {grad.shape} to 1D target {target_shape}")

    if len(target_shape) == 2:
        # It will always be unbroadcasting (R, C) back to (R, 1) or (1, C) or (1, 1). Otherwise the first check should caught it.
        axes_to_sum = []
        if target_shape[0] == 1 and grad.shape[0] > 1:
            axes_to_sum.append(0)
        if target_shape[1] == 1 and grad.shape[1] > 1:
            axes_to_sum.append(1)

        if axes_to_sum:
            sum_grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
            return sum_grad
        else:
            raise ValueError(f"Cannot unbroadcast 2D grad {grad.shape} to 2D target {target_shape}")


class Value:

    def __init__(self, data, _children=(), _op=''):
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got type {type(data)}")
            
        if data.ndim > 2:
            raise ValueError(f"Value class restricted to ndim <= 2, got shape {data.shape}")

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"shape={self.shape}\ndata={self.data}\ngrad_shape={self.grad.shape if hasattr(self, 'grad') else None}"

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other), op='+')

        def _backward():
            self.grad += unbroadcast_grad(self.shape, out.grad)
            other.grad += unbroadcast_grad(other.shape, out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):  # Element-wise product, a.k.a Hadamard's product
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast_grad(self.shape, other.data * out.grad)
            other.grad += unbroadcast_grad(other.shape, self.data * out.grad)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if self.data.ndim < 1 or other.data.ndim < 1:
            raise ValueError("Matmul requires operand with at lest 1 dimension")
        
        out_data = np.matmul(self.data, other.data)
        out = Value(out_data, (self, other), '@')
        
        def _backward():
            dL_dC = out.grad
            
            if self.ndim == 2 and other.ndim == 2:
                self.grad += np.matmul(dL_dC, other.data.T)
                other.grad += np.matmul(self.data.T, dL_dC)
            
            elif self.ndim == 2 and other.ndim == 1:
                self.grad += np.outer(dL_dC, other.data)
                other.grad += np.matmul(self.data.T, dL_dC)
            
            elif self.ndim == 1 and other.ndim == 2:
                self.grad += np.matmul(dL_dC, other.data.T)
                other.grad += np.outer(self.data, dL_dC)
            
            elif self.ndim == 1 and other.ndim == 1:
                self.grad += dL_dC * other.data
                other.grad += dL_dC * self.data
            
            else:
                raise RuntimeError(f"Unhandled ndim in matmul's backward: self={self.ndim}D, other={other.ndim}D")
            
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        if self.ndim == 0:
            return self
        
        if axis is not None and axis not in [0, 1]:
            raise ValueError(f"Axis must be either None, 0 or 1")
        if self.ndim == 1 and axis == 1:
            raise ValueError(f"Axis out of bound")
        
        out_data = np.sum(self.data, axis=axis, keepdims=True)
        out = Value(out_data, (self,), 'sum')
        
        def _backward():
            if axis is None:
                grad = np.ones_like(self.data) * out.grad
            else:
                if keepdims:
                    grad = out.grad
                else:
                    if self.ndim == 1:
                        grad = np.ones_like(self.data) * out.grad
                    elif self.ndim == 2:
                        if axis == 0:
                            grad = np.reshape(out.grad, (1, -1))
                        else:
                            grad = np.reshape(out.grad, (-1, 1))
            self.grad += grad
        
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
