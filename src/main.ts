import { LADO, preparar } from './preparo';
import { lerRede, prever, type Rede } from './rede';

interface Metricas { acuracia: number; imagensTeste: number; confusao: number[][] }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const quadro = $<HTMLCanvasElement>('quadro');
const ctx = quadro.getContext('2d', { willReadFrequently: true })!;
const visao = $<HTMLCanvasElement>('visao').getContext('2d')!;
const TRACO = quadro.width / 14; // grossura parecida com a dos dígitos do MNIST

const barras = Array.from({ length: 10 }, (_, d) => {
  const li = document.createElement('li');
  li.innerHTML = `<span>${d}</span><div><i></i></div><small></small>`;
  $('barras').append(li);
  return li;
});

let rede: Rede | null = null;
let agendado = false;

function tinta() {
  // O traço é desenhado sobre fundo transparente; a opacidade de cada pixel é a intensidade.
  const { data } = ctx.getImageData(0, 0, quadro.width, quadro.height);
  const t = new Float32Array(quadro.width * quadro.height);
  for (let i = 0; i < t.length; i++) t[i] = data[i * 4 + 3] / 255;
  return t;
}

function mostrarVisao(img: Float32Array | null) {
  const dados = visao.createImageData(LADO, LADO);
  img?.forEach((v, i) => {
    dados.data.fill(255, i * 4, i * 4 + 3);
    dados.data[i * 4 + 3] = v * 255;
  });
  visao.putImageData(dados, 0, 0);
}

function reconhecer() {
  agendado = false;
  const img = preparar(tinta(), quadro.width);
  mostrarVisao(img);
  $('dica').hidden = !!img;
  const p = img && rede ? prever(rede, img) : null;
  const melhor = p ? p.indexOf(Math.max(...p)) : -1;

  $('digito').textContent = p ? String(melhor) : '–';
  $('confianca').textContent = p ? `${Math.round(p[melhor] * 100)}% de confiança` : rede ? 'Aguardando o desenho' : 'Carregando a rede…';
  barras.forEach((li, d) => {
    const v = p?.[d] ?? 0;
    li.classList.toggle('melhor', d === melhor);
    li.querySelector('i')!.style.width = `${v * 100}%`;
    li.querySelector('small')!.textContent = p ? `${Math.round(v * 100)}%` : '';
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
  reconhecer();
});

ctx.lineWidth = TRACO;
ctx.lineCap = ctx.lineJoin = 'round';
reconhecer();

function mostrarMetricas({ acuracia, imagensTeste, confusao }: Metricas) {
  const pct = (v: number) => `${(v * 100).toLocaleString('pt-BR', { maximumFractionDigits: 2 })}%`;
  $('acuracia').textContent = pct(acuracia);
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
