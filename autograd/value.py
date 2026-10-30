import math

class Value:
    """
    Enveloppe un nombre scalaire 
    Construit dynamiquement le graphe d'exécution pour calculer les gradients par rétropropagation.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        # Fonction lambda exécutant la dérivée locale 
        self._backward = lambda: None
        # Enregistre les nœuds parents qui ont produit ce nœud
        self._prev = set(_children)
        # Type d'opération et étiquette de débogage
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- OPÉRATIONS ARITHMÉTIQUES DE BASE ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "L'exposant doit être un int ou un float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    # --- FONCTIONS D'ACTIVATION ET ÉLÉMENTAIRES POUR ATTENTION ET LOSS ---

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d/dx(exp(x)) = exp(x)
        out._backward = _backward

        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad  # d/dx(ln(x)) = 1/x
        out._backward = _backward

        return out

    # --- OPÉRATIONS MIROIRS ET RACCOURCIS ---

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return Value(other) + (-self)

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * (other ** -1)

    def __rtruediv__(self, other):  # other / self
        return Value(other) * (self ** -1)

    # --- MOTEUR DE RÉTROPROPAGATION ---

    def backward(self):
        """
        Ordonne le graphe acyclique orienté par tri topologique 
        et exécute la rétropropagation du résultat vers les feuilles.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # La dérivée de la sortie par rapport à elle-même est 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
