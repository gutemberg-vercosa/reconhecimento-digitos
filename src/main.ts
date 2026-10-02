import { LADO, separar } from './preparo';
import { lerRede, prever, type Rede } from './rede';

interface Metricas { acuracia: number; imagensTeste: number; confusao: number[][] }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const quadro = $<HTMLCanvasElement>('quadro');
const ctx = quadro.getContext('2d', { willReadFrequently: true })!;
const digitosEl = $('digitos');
const TRACO = quadro.height / 14; // grossura parecida com a dos dígitos do MNIST

const barras = Array.from({ length: 10 }, (_, d) => {
  const li = document.createElement('li');
  li.innerHTML = `<span>${d}</span><div><i></i></div><small></small>`;
  $('barras').append(li);
  return li;
});

let rede: Rede | null = null;
let agendado = false;
let selecionado = 0; // dígito do número cujas probabilidades aparecem nas barras

function tinta() {
  // O traço é desenhado sobre fundo transparente; a opacidade de cada pixel é a intensidade.
  const { data } = ctx.getImageData(0, 0, quadro.width, quadro.height);
  const t = new Float32Array(quadro.width * quadro.height);
  for (let i = 0; i < t.length; i++) t[i] = data[i * 4 + 3] / 255;
  return t;
}

/** Miniatura 28x28 do que a rede recebe: dígito branco sobre preto, como no MNIST. */
function miniatura(img: Float32Array) {
  const c = document.createElement('canvas');
  c.width = c.height = LADO;
  const g = c.getContext('2d')!;
  const dados = g.createImageData(LADO, LADO);
  img.forEach((v, i) => {
    dados.data.fill(255, i * 4, i * 4 + 3);
    dados.data[i * 4 + 3] = v * 255;
  });
  g.putImageData(dados, 0, 0);
  return c;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

function reconhecer() {
  agendado = false;
  const imgs = separar(tinta(), quadro.width, quadro.height);
  $('dica').hidden = imgs.length > 0;
  const resultados = rede ? imgs.map((img) => prever(rede!, img)) : [];
  const lidos = resultados.map((p) => p.indexOf(Math.max(...p)));
  selecionado = Math.min(selecionado, Math.max(0, lidos.length - 1));

  $('numero').textContent = lidos.length ? lidos.join('') : '–';
  $('confianca').textContent = !rede
    ? 'Carregando a rede…'
    : !lidos.length
      ? 'Aguardando o desenho'
      : lidos.length === 1
        ? `${pct(resultados[0][lidos[0]])} de confiança`
        : `${lidos.length} dígitos. Toque em um para ver as probabilidades.`;

  digitosEl.replaceChildren(...lidos.map((d, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(i === selecionado));
    b.setAttribute('aria-label', `Dígito ${i + 1}: ${d}, ${pct(resultados[i][d])}`);
    b.append(miniatura(imgs[i]));
    b.insertAdjacentHTML('beforeend', `<span><b>${d}</b> ${pct(resultados[i][d])}</span>`);
    b.addEventListener('click', () => { selecionado = i; reconhecer(); });
    return b;
  }));

  const p = resultados[selecionado];
  barras.forEach((li, d) => {
    li.classList.toggle('melhor', d === lidos[selecionado]);
    li.querySelector('i')!.style.width = `${(p?.[d] ?? 0) * 100}%`;
    li.querySelector('small')!.textContent = p ? pct(p[d]) : '';
  });
}

const agendar = () => {
  if (!agendado) requestAnimationFrame(reconhecer);
  agendado = true;
};

// Desenho com mouse, dedo ou caneta; as coordenadas da tela são convertidas para as do canvas.
function ponto(e: PointerEvent) {
  const r = quadro.getBoundingClientRect();
  return [((e.clientX - r.left) * quadro.width) / r.width, ((e.clientY - r.top) * quadro.height) / r.height];
}

quadro.addEventListener('pointerdown', (e) => {
  quadro.setPointerCapture(e.pointerId);
  const [x, y] = ponto(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y); // um toque sem arrastar também marca um ponto
  ctx.stroke();
  agendar();
});

quadro.addEventListener('pointermove', (e) => {
  if (!quadro.hasPointerCapture(e.pointerId)) return;
  const [x, y] = ponto(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  agendar();
});

$('limpar').addEventListener('click', () => {
  ctx.clearRect(0, 0, quadro.width, quadro.height);
  selecionado = 0;
  reconhecer();
});

ctx.lineWidth = TRACO;
ctx.lineCap = ctx.lineJoin = 'round';
reconhecer();

function mostrarMetricas({ acuracia, imagensTeste, confusao }: Metricas) {
  $('acuracia').textContent = `${(acuracia * 100).toLocaleString('pt-BR', { maximumFractionDigits: 2 })}%`;
  $('teste').textContent = imagensTeste.toLocaleString('pt-BR');
  const erros = confusao
    .flatMap((linha, real) => linha.map((n, lido) => ({ real, lido, n })))
    .filter((e) => e.real !== e.lido)
    .sort((a, b) => b.n - a.n)
    .slice(0, 3);
  $('confusoes').innerHTML = erros.map((e) => `<li><b>${e.real}</b> lido como <b>${e.lido}</b>: ${e.n} vezes</li>`).join('');
  $('metricas').hidden = false;
}

Promise.all([fetch('modelo.bin').then((r) => r.arrayBuffer()), fetch('metricas.json').then((r) => r.json())])
  .then(([pesos, metricas]) => {
    rede = lerRede(pesos);
    mostrarMetricas(metricas);
    reconhecer();
  })
  .catch(() => {
    $('confianca').textContent = 'Não foi possível carregar a rede. Recarregue a página.';
  });
