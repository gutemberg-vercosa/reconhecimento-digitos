// Separa o desenho em dígitos e deixa cada um no formato das imagens do MNIST: o dígito
// cabe numa área de 20x20, dentro de uma imagem 28x28, com o centro de massa no meio.
export const LADO = 28;
const AREA_DIGITO = 20;
const TRACO_MNIST = 0.14; // grossura do traço em relação ao lado maior do dígito, como no MNIST
const MENOR_DIGITO = 0.1; // fração da altura do quadro; abaixo disso é um ponto solto, não um dígito

interface Grupo { x0: number; y0: number; x1: number; y1: number; rotulos: Set<number> }

const larg = (g: Grupo) => g.x1 - g.x0 + 1;
const alt = (g: Grupo) => g.y1 - g.y0 + 1;
const sobreposicao = (a0: number, a1: number, b0: number, b1: number) => Math.min(a1, b1) - Math.max(a0, b0) + 1;

/**
 * `tinta` é um quadro largura x altura com a intensidade de cada pixel (0 a 1).
 * 	raco é a grossura do pincel, usada para afinar os dígitos pequenos.
 * Devolve as linhas do desenho, de cima para baixo, e em cada uma as imagens 28x28
 * dos dígitos, da esquerda para a direita.
 */
export function separar(tinta: ArrayLike<number>, largura: number, altura: number, traco = 0): Float32Array[][] {
  const { rotulo, grupos } = rotular(tinta, largura, altura);

  // Traços que não se tocam, mas estão na mesma coluna, são o mesmo dígito quando se
  // sobrepõem na vertical (um 4 feito em dois traços) ou quando um deles é pequeno e
  // está quase encostado no outro (o corte de um 5). Dois dígitos inteiros, um em cima
  // do outro, ficam separados, mesmo próximos.
  const pedacos = [...grupos];
  for (let i = 0; i < pedacos.length; i++) {
    for (let j = i + 1; j < pedacos.length; j++) {
      const a = pedacos[i], b = pedacos[j];
      const mesmaColuna = sobreposicao(a.x0, a.x1, b.x0, b.x1) >= Math.min(larg(a), larg(b)) / 2;
      const vertical = sobreposicao(a.y0, a.y1, b.y0, b.y1);
      const maior = Math.max(alt(a), alt(b));
      const pedacoPequeno = Math.min(alt(a), alt(b)) < maior * 0.4 && -vertical <= maior * 0.3;
      if (!mesmaColuna || (vertical <= 0 && !pedacoPequeno)) continue;
      a.x0 = Math.min(a.x0, b.x0); a.x1 = Math.max(a.x1, b.x1);
      a.y0 = Math.min(a.y0, b.y0); a.y1 = Math.max(a.y1, b.y1);
      b.rotulos.forEach((r) => a.rotulos.add(r));
      pedacos.splice(j, 1);
      j = i; // o grupo cresceu: confere de novo os que restam
    }
  }

  // Dígitos que dividem a maior parte da altura estão na mesma linha.
  const linhas: Grupo[][] = [];
  const digitos = pedacos
    .filter((d) => Math.max(larg(d), alt(d)) >= altura * MENOR_DIGITO)
    .sort((a, b) => a.y0 + a.y1 - (b.y0 + b.y1));
  for (const d of digitos) {
    const linha = linhas.find((l) => l.some((o) => sobreposicao(o.y0, o.y1, d.y0, d.y1) >= Math.min(alt(o), alt(d)) / 2));
    if (linha) linha.push(d);
    else linhas.push([d]);
  }

  return linhas.map((l) => l
    .sort((a, b) => a.x0 - b.x0)
    .map((d) => reduzir(afinar(recortar(tinta, largura, d, rotulo), traco))));
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

interface Recorte { w: number; h: number; v: Float32Array }

/** Copia só a tinta do dígito, dentro da caixa dele. */
function recortar(tinta: ArrayLike<number>, largura: number, d: Grupo, rotulo: Int32Array): Recorte {
  const w = larg(d), h = alt(d), v = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (d.y0 + y) * largura + d.x0 + x;
      if (d.rotulos.has(rotulo[i])) v[y * w + x] = tinta[i];
    }
  }
  return { w, h, v };
}

/**
 * O traço tem a mesma grossura em qualquer dígito, então os pequenos (numa linha de
 * baixo, por exemplo) ficam grossos demais e viram borrões. Aqui o traço é afinado por
 * erosão até a proporção das imagens do MNIST.
 */
function afinar(r: Recorte, traco: number): Recorte {
  const raio = Math.min(Math.floor((traco - Math.max(r.w, r.h) * TRACO_MNIST) / 2), Math.floor(traco / 2) - 2);
  if (raio < 1) return r;
  // Erosão quadrada em duas passadas: cada pixel fica com o menor valor ao redor.
  const minimo = (v: Float32Array, passo: number, n: number, pos: (i: number) => number) =>
    v.map((_, i) => {
      let m = 1;
      for (let k = -raio; k <= raio && m; k++) m = pos(i) + k < 0 || pos(i) + k >= n ? 0 : Math.min(m, v[i + k * passo]);
      return m;
    });
  const v = minimo(minimo(r.v, 1, r.w, (i) => i % r.w), r.w, r.h, (i) => Math.floor(i / r.w));

  // A erosão encolhe o desenho: recorta de novo a caixa do que sobrou.
  let x0 = r.w, y0 = r.h, x1 = -1, y1 = -1;
  v.forEach((t, i) => {
    if (!t) return;
    const x = i % r.w, y = Math.floor(i / r.w);
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  });
  if (x1 < 0) return r;
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  return { w, h, v: Float32Array.from({ length: w * h }, (_, i) => v[(y0 + Math.floor(i / w)) * r.w + x0 + (i % w)]) };
}

/**
 * Reduz o recorte para caber em 20x20, mantendo a proporção. Cada pixel de origem
 * soma sua parte na área do pixel de destino, o que equivale a uma média.
 */
function reduzir({ w, h, v }: Recorte) {
  const escala = AREA_DIGITO / Math.max(w, h);
  const dx = (LADO - w * escala) / 2;
  const dy = (LADO - h * escala) / 2;
  const img = new Float32Array(LADO * LADO);
  v.forEach((t, i) => {
    if (!t) return;
    const tx = Math.floor(dx + ((i % w) + 0.5) * escala);
    const ty = Math.floor(dy + (Math.floor(i / w) + 0.5) * escala);
    img[ty * LADO + tx] += t * escala * escala;
  });
  return centralizar(img.map((t) => Math.min(1, t)));
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
