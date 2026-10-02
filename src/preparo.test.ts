import { describe, expect, it } from 'vitest';
import { centralizar, LADO, separar } from './preparo';

const L = 200, A = 100;

// Quadro 200x100 com retângulos de tinta [x0, y0, x1, y1).
function quadro(...retangulos: number[][]) {
  const tinta = new Float32Array(L * A);
  for (const [x0, y0, x1, y1] of retangulos) {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) tinta[y * L + x] = 1;
  }
  return tinta;
}

const centroDeMassa = (img: Float32Array) => {
  let m = 0, x = 0, y = 0;
  img.forEach((v, i) => { m += v; x += v * (i % LADO); y += v * Math.floor(i / LADO); });
  return [x / m, y / m];
};

const linhasComTinta = (img: Float32Array) => new Set([...img.keys()].filter((i) => img[i]).map((i) => Math.floor(i / LADO))).size;

describe('separar', () => {
  it('não acha dígitos num quadro vazio', () => {
    expect(separar(quadro(), L, A)).toEqual([]);
  });

  it('ajusta o dígito a 20 pixels de altura e o centraliza', () => {
    const [[img]] = separar(quadro([5, 5, 13, 45]), L, A);
    expect(linhasComTinta(img)).toBe(20);
    const [x, y] = centroDeMassa(img);
    expect(Math.abs(x - 13.5)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(y - 13.5)).toBeLessThanOrEqual(0.5);
    expect(Math.max(...img)).toBeLessThanOrEqual(1);
  });

  it('separa traços lado a lado, da esquerda para a direita', () => {
    // Um traço alto à direita e um baixo à esquerda: a ordem segue o eixo x.
    const [imgs] = separar(quadro([150, 10, 160, 90], [20, 50, 60, 90]), L, A);
    expect(imgs).toHaveLength(2);
    const largura = (img: Float32Array) => new Set([...img.keys()].filter((i) => img[i]).map((i) => i % LADO)).size;
    expect(largura(imgs[0])).toBe(20); // o retângulo largo ocupa os 20 pixels na horizontal
  });

  it('junta traços soltos na mesma coluna num só dígito', () => {
    // Como o corte de cima de um 5, que não encosta no resto.
    expect(separar(quadro([40, 10, 80, 18], [40, 25, 80, 90]), L, A).flat()).toHaveLength(1);
  });

  it('ignora pontos soltos pequenos demais para serem dígitos', () => {
    expect(separar(quadro([10, 10, 60, 90], [150, 50, 155, 55]), L, A).flat()).toHaveLength(1);
  });

  it('lê um dígito logo abaixo de outro como outra linha, não como o mesmo dígito', () => {
    // Duas linhas de dois dígitos; os de baixo ficam na mesma coluna dos de cima.
    const linhas = separar(quadro([20, 5, 40, 40], [80, 5, 100, 40], [20, 55, 40, 95], [80, 55, 100, 95]), L, A);
    expect(linhas.map((l) => l.length)).toEqual([2, 2]);
  });

  it('separa dígitos empilhados mesmo quando as linhas estão bem próximas', () => {
    const linhas = separar(quadro([20, 5, 40, 47], [20, 52, 40, 95]), L, A);
    expect(linhas.map((l) => l.length)).toEqual([1, 1]);
  });});

describe('centralizar', () => {
  it('leva um ponto do canto para o meio', () => {
    const img = new Float32Array(LADO * LADO);
    img[0] = 1;
    expect(centralizar(img).indexOf(1)).toBe(14 * LADO + 14);
  });
});
