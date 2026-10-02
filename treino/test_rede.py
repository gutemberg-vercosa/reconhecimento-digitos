import unittest

import numpy as np

from rede import Adam, iniciar, perda_e_gradientes, prever


class TestRede(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Rede pequena em float64 para a conferência numérica ser precisa.
        self.p = {k: v.astype(np.float64) for k, v in iniciar(6, 5, 3, rng).items()}
        self.x = rng.normal(size=(4, 6))
        self.y = np.array([0, 2, 1, 2])

    def test_probabilidades_somam_um(self):
        _, prob = prever(self.p, self.x)
        np.testing.assert_allclose(prob.sum(axis=1), 1)

    def test_gradientes_conferem_com_diferencas_finitas(self):
        _, grad = perda_e_gradientes(self.p, self.x, self.y)
        eps = 1e-6
        for k, v in self.p.items():
            numerico = np.zeros_like(v)
            for i in np.ndindex(v.shape):
                original = v[i]
                v[i] = original + eps
                mais, _ = perda_e_gradientes(self.p, self.x, self.y)
                v[i] = original - eps
                menos, _ = perda_e_gradientes(self.p, self.x, self.y)
                v[i] = original
                numerico[i] = (mais - menos) / (2 * eps)
            np.testing.assert_allclose(grad[k], numerico, atol=1e-7, err_msg=k)

    def test_adam_reduz_a_perda(self):
        adam = Adam(self.p, taxa=1e-2)
        inicial, _ = perda_e_gradientes(self.p, self.x, self.y)
        for _ in range(100):
            _, grad = perda_e_gradientes(self.p, self.x, self.y)
            adam.passo(self.p, grad)
        final, _ = perda_e_gradientes(self.p, self.x, self.y)
        self.assertLess(final, inicial / 10)


if __name__ == '__main__':
    unittest.main()
