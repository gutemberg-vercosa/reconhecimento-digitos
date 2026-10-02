// Deixa o desenho no formato das imagens do MNIST: o dígito cabe numa área de 20x20,
// dentro de uma imagem 28x28, com o centro de massa no meio.
export const LADO = 28;
const AREA_DIGITO = 20;

/** `tinta` é um quadro lado x lado com a intensidade de cada pixel (0 a 1). Devolve null se estiver vazio. */
export function preparar(tinta: ArrayLike<number>, lado: number): Float32Array | null {
  let x0 = lado, y0 = lado, x1 = -1, y1 = -1;
  for (let i = 0; i < lado * lado; i++) {
    if (!tinta[i]) continue;
    const x = i % lado, y = Math.floor(i / lado);
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  if (x1 < 0) return null;

  // Reduz a caixa do desenho para caber em 20x20, mantendo a proporção. Cada pixel de
  // origem soma sua parte na área do pixel de destino, o que equivale a uma média.
  const escala = AREA_DIGITO / Math.max(x1 - x0 + 1, y1 - y0 + 1);
  const dx = (LADO - (x1 - x0 + 1) * escala) / 2;
  const dy = (LADO - (y1 - y0 + 1) * escala) / 2;
  const img = new Float32Array(LADO * LADO);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const v = tinta[y * lado + x];
      if (!v) continue;
      const tx = Math.floor(dx + (x - x0 + 0.5) * escala);
      const ty = Math.floor(dy + (y - y0 + 0.5) * escala);
      img[ty * LADO + tx] += v * escala * escala;
    }
  }
  return centralizar(img.map((v) => Math.min(1, v)));
}

/** Desloca a imagem para o centro de massa cair no meio, como no MNIST. */
export function centralizar(img: Float32Array): Float32Array {
  let massa = 0, mx = 0, my = 0;
  img.forEach((v, i) => {
    massa += v;
    mx += v * (i % LADO);
    my += v * Math.floor(i / LADO);
  });
  if (!massa) return img;
  const meio = (LADO - 1) / 2;
  const sx = Math.round(meio - mx / massa), sy = Math.round(meio - my / massa);
  const saida = new Float32Array(LADO * LADO);
  img.forEach((v, i) => {
    const x = (i % LADO) + sx, y = Math.floor(i / LADO) + sy;
    if (v && x >= 0 && x < LADO && y >= 0 && y < LADO) saida[y * LADO + x] = v;
  });
  return saida;
}
