// Separa o desenho em dígitos e deixa cada um no formato das imagens do MNIST: o dígito
// cabe numa área de 20x20, dentro de uma imagem 28x28, com o centro de massa no meio.
export const LADO = 28;
const AREA_DIGITO = 20;
const MENOR_DIGITO = 0.1; // fração da altura do quadro; abaixo disso é um ponto solto, não um dígito

interface Grupo { x0: number; y0: number; x1: number; y1: number; rotulos: Set<number> }

/**
 * `tinta` é um quadro largura x altura com a intensidade de cada pixel (0 a 1).
 * Devolve uma imagem 28x28 por dígito, da esquerda para a direita.
 */
export function separar(tinta: ArrayLike<number>, largura: number, altura: number): Float32Array[] {
  const { rotulo, grupos } = rotular(tinta, largura, altura);

  // Traços que não se tocam mas ficam na mesma coluna (o corte de um 5, o 4 feito em
  // dois traços) pertencem ao mesmo dígito.
  const digitos: Grupo[] = [];
  for (const g of grupos.sort((a, b) => a.x0 - b.x0)) {
    const ant = digitos.at(-1);
    if (ant && Math.min(ant.x1, g.x1) - g.x0 + 1 >= Math.min(ant.x1 - ant.x0, g.x1 - g.x0) / 2) {
      ant.x0 = Math.min(ant.x0, g.x0); ant.x1 = Math.max(ant.x1, g.x1);
      ant.y0 = Math.min(ant.y0, g.y0); ant.y1 = Math.max(ant.y1, g.y1);
      g.rotulos.forEach((r) => ant.rotulos.add(r));
    } else digitos.push(g);
  }

  return digitos
    .filter((d) => Math.max(d.x1 - d.x0, d.y1 - d.y0) + 1 >= altura * MENOR_DIGITO)
    .map((d) => reduzir(tinta, largura, d, (i) => d.rotulos.has(rotulo[i])));
}

/** Rotula os grupos de pixels com tinta que se tocam (vizinhança de 8). */
function rotular(tinta: ArrayLike<number>, largura: number, altura: number) {
  const rotulo = new Int32Array(largura * altura).fill(-1);
  const grupos: Grupo[] = [];
  for (let inicio = 0; inicio < rotulo.length; inicio++) {
    if (!tinta[inicio] || rotulo[inicio] >= 0) continue;
    const r = grupos.length;
    const g: Grupo = { x0: largura, y0: altura, x1: -1, y1: -1, rotulos: new Set([r]) };
    const pilha = [inicio];
    rotulo[inicio] = r;
    while (pilha.length) {
      const i = pilha.pop()!;
      const x = i % largura, y = Math.floor(i / largura);
      g.x0 = Math.min(g.x0, x); g.x1 = Math.max(g.x1, x);
      g.y0 = Math.min(g.y0, y); g.y1 = Math.max(g.y1, y);
      for (let vy = Math.max(0, y - 1); vy <= Math.min(altura - 1, y + 1); vy++) {
        for (let vx = Math.max(0, x - 1); vx <= Math.min(largura - 1, x + 1); vx++) {
          const v = vy * largura + vx;
          if (tinta[v] && rotulo[v] < 0) { rotulo[v] = r; pilha.push(v); }
        }
      }
    }
    grupos.push(g);
  }
  return { rotulo, grupos };
}

/**
 * Reduz a caixa do dígito para caber em 20x20, mantendo a proporção. Cada pixel de
 * origem soma sua parte na área do pixel de destino, o que equivale a uma média.
 */
function reduzir(tinta: ArrayLike<number>, largura: number, d: Grupo, pertence: (i: number) => boolean) {
  const escala = AREA_DIGITO / Math.max(d.x1 - d.x0 + 1, d.y1 - d.y0 + 1);
  const dx = (LADO - (d.x1 - d.x0 + 1) * escala) / 2;
  const dy = (LADO - (d.y1 - d.y0 + 1) * escala) / 2;
  const img = new Float32Array(LADO * LADO);
  for (let y = d.y0; y <= d.y1; y++) {
    for (let x = d.x0; x <= d.x1; x++) {
      const i = y * largura + x;
      if (!tinta[i] || !pertence(i)) continue;
      const tx = Math.floor(dx + (x - d.x0 + 0.5) * escala);
      const ty = Math.floor(dy + (y - d.y0 + 0.5) * escala);
      img[ty * LADO + tx] += tinta[i] * escala * escala;
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
