import { describe, expect, it } from 'vitest';
import { centralizar, LADO, preparar } from './preparo';

// Um traço vertical num canto de um quadro 100x100.
function traco() {
  const tinta = new Float32Array(100 * 100);
  for (let y = 5; y < 45; y++) for (let x = 5; x < 13; x++) tinta[y * 100 + x] = 1;
  return tinta;
}

const centroDeMassa = (img: Float32Array) => {
  let m = 0, x = 0, y = 0;
  img.forEach((v, i) => { m += v; x += v * (i % LADO); y += v * Math.floor(i / LADO); });
  return [x / m, y / m];
};

describe('preparar', () => {
  it('devolve null para um quadro vazio', () => {
    expect(preparar(new Float32Array(100), 10)).toBeNull();
  });

  it('ajusta o desenho a 20 pixels de altura e o centraliza', () => {
    const img = preparar(traco(), 100)!;
    const linhas = new Set([...img.keys()].filter((i) => img[i]).map((i) => Math.floor(i / LADO)));
    expect(linhas.size).toBe(20);
    const [x, y] = centroDeMassa(img);
    expect(Math.abs(x - 13.5)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(y - 13.5)).toBeLessThanOrEqual(0.5);
    expect(Math.max(...img)).toBeLessThanOrEqual(1);
  });
});

describe('centralizar', () => {
  it('leva um ponto do canto para o meio', () => {
    const img = new Float32Array(LADO * LADO);
    img[0] = 1;
    const saida = centralizar(img);
    expect(saida.indexOf(1)).toBe(14 * LADO + 14);
  });
});
