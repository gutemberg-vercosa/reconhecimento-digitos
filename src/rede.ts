// Mesma rede treinada em treino/rede.py: 784 -> 128 (ReLU) -> 10 (softmax).
export const ENTRADAS = 784;
export const OCULTOS = 128;
export const SAIDAS = 10;

export interface Rede {
  W1: Float32Array; // ENTRADAS x OCULTOS, linha por linha
  b1: Float32Array;
  W2: Float32Array; // OCULTOS x SAIDAS
  b2: Float32Array;
}

/** Lê o modelo.bin exportado pelo treino: W1, b1, W2 e b2 em float32, nessa ordem. */
export function lerRede(buffer: ArrayBuffer): Rede {
  const valores = new Float32Array(buffer);
  let i = 0;
  const pedaco = (n: number) => valores.subarray(i, (i += n));
  const rede = { W1: pedaco(ENTRADAS * OCULTOS), b1: pedaco(OCULTOS), W2: pedaco(OCULTOS * SAIDAS), b2: pedaco(SAIDAS) };
  if (i !== valores.length) throw new Error('Arquivo de pesos com tamanho inesperado');
  return rede;
}

/** Probabilidade de cada dígito para uma imagem 28x28 com valores de 0 a 1. */
export function prever({ W1, b1, W2, b2 }: Rede, x: ArrayLike<number>): number[] {
  const h = Float32Array.from(b1);
  for (let i = 0; i < ENTRADAS; i++) {
    if (!x[i]) continue; // a maior parte da imagem é fundo
    for (let j = 0; j < OCULTOS; j++) h[j] += x[i] * W1[i * OCULTOS + j];
  }
  const z = Array.from(b2);
  for (let j = 0; j < OCULTOS; j++) {
    if (h[j] <= 0) continue; // ReLU
    for (let k = 0; k < SAIDAS; k++) z[k] += h[j] * W2[j * SAIDAS + k];
  }
  const maior = Math.max(...z);
  const e = z.map((v) => Math.exp(v - maior));
  const soma = e.reduce((a, b) => a + b);
  return e.map((v) => v / soma);
}
