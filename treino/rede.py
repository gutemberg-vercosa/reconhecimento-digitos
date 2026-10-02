"""Rede neural de duas camadas (784 -> 128 -> 10) escrita só com NumPy."""
import numpy as np


def iniciar(entradas, ocultos, saidas, rng):
    # Inicialização de He: a escala certa para camadas com ReLU.
    return {
        'W1': rng.normal(0, np.sqrt(2 / entradas), (entradas, ocultos)).astype(np.float32),
        'b1': np.zeros(ocultos, np.float32),
        'W2': rng.normal(0, np.sqrt(2 / ocultos), (ocultos, saidas)).astype(np.float32),
        'b2': np.zeros(saidas, np.float32),
    }


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def prever(p, x):
    """Devolve a camada oculta e a probabilidade de cada dígito."""
    h = np.maximum(0, x @ p['W1'] + p['b1'])
    return h, softmax(h @ p['W2'] + p['b2'])


def perda_e_gradientes(p, x, y):
    """Entropia cruzada média e os gradientes de cada parâmetro; y são os rótulos (0 a 9)."""
    n = len(x)
    h, prob = prever(p, x)
    perda = -np.log(prob[np.arange(n), y] + 1e-12).mean()
    d2 = prob.copy()
    d2[np.arange(n), y] -= 1
    d2 /= n
    dh = (d2 @ p['W2'].T) * (h > 0)
    return perda, {'W1': x.T @ dh, 'b1': dh.sum(0), 'W2': h.T @ d2, 'b2': d2.sum(0)}


class Adam:
    def __init__(self, p, taxa=1e-3, b1=0.9, b2=0.999):
        self.taxa, self.b1, self.b2, self.t = taxa, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}

    def passo(self, p, grad):
        self.t += 1
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grad[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grad[k] ** 2
            m = self.m[k] / (1 - self.b1 ** self.t)
            v = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= (self.taxa * m / (np.sqrt(v) + 1e-8)).astype(p[k].dtype)
