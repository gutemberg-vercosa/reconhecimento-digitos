import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import amostras from './amostras.json';
import { lerRede, prever } from './rede';

const rede = lerRede(new Uint8Array(readFileSync('public/modelo.bin')).buffer);

describe('rede', () => {
  it('dá as mesmas probabilidades que o treino em Python', () => {
    for (const { pixels, probabilidades } of amostras) {
      const p = prever(rede, pixels.map((v) => v / 255));
      p.forEach((v, i) => expect(v).toBeCloseTo(probabilidades[i], 4));
    }
  });

  it('recusa um arquivo de pesos com tamanho errado', () => {
    expect(() => lerRede(new ArrayBuffer(16))).toThrow();
  });
});
