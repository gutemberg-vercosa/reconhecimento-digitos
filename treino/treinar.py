"""Treina a rede no MNIST e exporta os pesos e as métricas para o site.

Uso: python treino/treinar.py
"""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

from rede import Adam, iniciar, perda_e_gradientes, prever

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / 'treino' / 'dados'
ESPELHO = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
OCULTOS, EPOCAS, LOTE, DESLOCAMENTO = 128, 20, 128, 2


def carregar(nome):
    arquivo = DADOS / nome
    if not arquivo.exists():
        DADOS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ESPELHO + nome, arquivo)
    with gzip.open(arquivo) as f:
        dados = f.read()
    # Formato IDX: cabeçalho de 16 bytes nas imagens e de 8 nos rótulos.
    if 'images' in nome:
        return np.frombuffer(dados, np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255
    return np.frombuffer(dados, np.uint8, offset=8).astype(np.int64)


def deslocar(x, rng):
    """Move cada imagem do lote alguns pixels, para a rede tolerar dígitos fora do centro."""
    imgs = x.reshape(-1, 28, 28)
    dy, dx = rng.integers(-DESLOCAMENTO, DESLOCAMENTO + 1, 2)
    return np.roll(imgs, (dy, dx), axis=(1, 2)).reshape(-1, 784)


def main():
    rng = np.random.default_rng(42)
    x_treino, y_treino = carregar('train-images-idx3-ubyte.gz'), carregar('train-labels-idx1-ubyte.gz')
    x_teste, y_teste = carregar('t10k-images-idx3-ubyte.gz'), carregar('t10k-labels-idx1-ubyte.gz')

    p = iniciar(784, OCULTOS, 10, rng)
    adam = Adam(p)
    for epoca in range(1, EPOCAS + 1):
        ordem = rng.permutation(len(x_treino))
        for i in range(0, len(ordem), LOTE):
            lote = ordem[i:i + LOTE]
            _, grad = perda_e_gradientes(p, deslocar(x_treino[lote], rng), y_treino[lote])
            adam.passo(p, grad)
        acerto = (prever(p, x_teste)[1].argmax(1) == y_teste).mean()
        print(f'época {epoca:2}: acurácia no teste {acerto:.2%}')

    previsto = prever(p, x_teste)[1].argmax(1)
    confusao = np.zeros((10, 10), int)
    np.add.at(confusao, (y_teste, previsto), 1)

    # Pesos em float32 little-endian, na ordem que o site lê: W1, b1, W2, b2.
    (RAIZ / 'public' / 'modelo.bin').write_bytes(
        b''.join(p[k].astype('<f4').tobytes() for k in ('W1', 'b1', 'W2', 'b2')))
    (RAIZ / 'public' / 'metricas.json').write_text(json.dumps({
        'acuracia': float((previsto == y_teste).mean()),
        'imagensTreino': len(x_treino),
        'imagensTeste': len(x_teste),
        'confusao': confusao.tolist(),
    }))
    # Algumas imagens de teste com as probabilidades calculadas aqui, para os testes do
    # site conferirem que a inferência em TypeScript dá o mesmo resultado.
    (RAIZ / 'src' / 'amostras.json').write_text(json.dumps([
        {'pixels': np.round(x_teste[i] * 255).astype(int).tolist(),
         'probabilidades': prever(p, x_teste[i:i + 1])[1][0].round(6).tolist()}
        for i in range(5)
    ]))


if __name__ == '__main__':
    main()
