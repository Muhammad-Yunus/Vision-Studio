const sampleCache: Record<string, ImageData> = {};
const samplePromiseCache: Record<string, Promise<ImageData>> = {};

export const DEFAULT_LENA_URL = '/samples/lena.jpg';

function createCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  return [canvas, ctx];
}

function drawFallbackPortrait(w = 320, h = 320): ImageData {
  const [, ctx] = createCanvas(w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#462635');
  bg.addColorStop(0.5, '#7c444d');
  bg.addColorStop(1, '#2f1924');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function drawCalibrationSample(w = 320, h = 320): ImageData {
  const [, ctx] = createCanvas(w, h);
  ctx.fillStyle = '#181c24';
  ctx.fillRect(0, 0, w, h);

  const cell = 32;
  for (let y = 0; y < h - 72; y += cell) {
    for (let x = 0; x < w; x += cell) {
      const dark = (x / cell + y / cell) % 2 === 0;
      ctx.fillStyle = dark ? '#101318' : '#e5e7eb';
      ctx.fillRect(x, y, cell, cell);
    }
  }

  const colors = ['#ef4444', '#f59e0b', '#22c55e', '#06b6d4', '#3b82f6', '#a855f7', '#ffffff', '#111827'];
  const bw = w / colors.length;
  colors.forEach((c, idx) => {
    ctx.fillStyle = c;
    ctx.fillRect(idx * bw, h - 72, bw, 72);
  });

  ctx.strokeStyle = '#f58220';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(w / 2, (h - 72) / 2, 58, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(w / 2, (h - 72) / 2, 24, 0, Math.PI * 2);
  ctx.stroke();

  return ctx.getImageData(0, 0, w, h);
}

function drawPCBSample(w = 320, h = 320): ImageData {
  const [, ctx] = createCanvas(w, h);
  ctx.fillStyle = '#0c3b2a';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#26a670';
  ctx.lineWidth = 3;
  for (let i = 0; i < 14; i++) {
    const y = 24 + i * 20;
    ctx.beginPath();
    ctx.moveTo(16, y);
    ctx.lineTo(90 + (i % 4) * 20, y);
    ctx.lineTo(140 + (i % 3) * 18, y + 18);
    ctx.lineTo(w - 20, y + 18);
    ctx.stroke();
  }

  ctx.fillStyle = '#181a1f';
  ctx.fillRect(104, 92, 112, 112);
  ctx.strokeStyle = '#9ca3af';
  ctx.lineWidth = 2;
  ctx.strokeRect(104, 92, 112, 112);

  ctx.fillStyle = '#d1d5db';
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(114 + i * 12, 82, 6, 10);
    ctx.fillRect(114 + i * 12, 204, 6, 10);
    ctx.fillRect(94, 102 + i * 12, 10, 6);
    ctx.fillRect(216, 102 + i * 12, 10, 6);
  }

  ctx.fillStyle = '#f59e0b';
  [
    [42, 48],
    [276, 48],
    [42, 272],
    [276, 272],
  ].forEach(([cx, cy]) => {
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0c3b2a';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#f59e0b';
  });

  ctx.fillStyle = '#b45309';
  ctx.fillRect(36, 130, 34, 18);
  ctx.fillRect(248, 150, 36, 20);
  ctx.fillStyle = '#38bdf8';
  ctx.fillRect(130, 244, 54, 24);

  return ctx.getImageData(0, 0, w, h);
}

function drawUrbanSample(w = 320, h = 320): ImageData {
  const [, ctx] = createCanvas(w, h);
  const sky = ctx.createLinearGradient(0, 0, 0, h * 0.56);
  sky.addColorStop(0, '#0f172a');
  sky.addColorStop(0.65, '#1e3a5f');
  sky.addColorStop(1, '#f97316');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h * 0.56);

  ctx.fillStyle = '#111827';
  const buildings = [
    [0, 105, 42, 75],
    [44, 72, 38, 108],
    [84, 95, 48, 85],
    [138, 60, 44, 120],
    [192, 85, 52, 95],
    [250, 68, 70, 112],
  ];
  buildings.forEach(([x, y, bw, bh]) => ctx.fillRect(x, y, bw, bh));

  ctx.fillStyle = '#1f2937';
  ctx.fillRect(0, h * 0.56, w, h * 0.44);

  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 4;
  ctx.setLineDash([16, 12]);
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h * 0.58);
  ctx.lineTo(w * 0.5, h);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(w * 0.42, h * 0.56);
  ctx.lineTo(w * 0.12, h);
  ctx.moveTo(w * 0.58, h * 0.56);
  ctx.lineTo(w * 0.88, h);
  ctx.stroke();

  ctx.fillStyle = '#38bdf8';
  ctx.fillRect(w * 0.36, h * 0.62, 78, 44);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(w * 0.39, h * 0.65, 58, 16);
  ctx.fillStyle = '#ef4444';
  ctx.fillRect(w * 0.37, h * 0.72, 12, 7);
  ctx.fillRect(w * 0.55, h * 0.72, 12, 7);

  return ctx.getImageData(0, 0, w, h);
}

export async function loadImageDataFromDataUrl(dataUrl: string, maxDim = 360): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const w = Math.max(64, Math.round(img.width * scale));
      const h = Math.max(64, Math.round(img.height * scale));
      const [, ctx] = createCanvas(w, h);
      ctx.drawImage(img, 0, 0, w, h);
      resolve(ctx.getImageData(0, 0, w, h));
    };
    img.onerror = () => reject(new Error(`Failed to decode image: ${dataUrl}`));
    img.src = dataUrl;
  });
}

export async function getSampleImageDataAsync(sampleId: string): Promise<ImageData> {
  const key = sampleId === 'lena' ? 'portrait' : sampleId || 'portrait';
  if (sampleCache[key]) {
    const src = sampleCache[key];
    return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
  }

  if (key === 'portrait') {
    if (!samplePromiseCache[key]) {
      samplePromiseCache[key] = loadImageDataFromDataUrl(DEFAULT_LENA_URL, 360)
        .then((img) => {
          sampleCache[key] = img;
          return img;
        })
        .catch(() => {
          const fb = drawFallbackPortrait();
          sampleCache[key] = fb;
          return fb;
        });
    }
    const loaded = await samplePromiseCache[key];
    return new ImageData(new Uint8ClampedArray(loaded.data), loaded.width, loaded.height);
  }

  if (key === 'calibration') sampleCache[key] = drawCalibrationSample();
  else if (key === 'pcb') sampleCache[key] = drawPCBSample();
  else sampleCache[key] = drawUrbanSample();

  const src = sampleCache[key];
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}

export function getSampleImageData(sampleId: string): ImageData {
  const key = sampleId === 'lena' ? 'portrait' : sampleId || 'portrait';
  if (!sampleCache[key]) {
    if (key === 'calibration') sampleCache[key] = drawCalibrationSample();
    else if (key === 'pcb') sampleCache[key] = drawPCBSample();
    else if (key === 'urban') sampleCache[key] = drawUrbanSample();
    else return drawFallbackPortrait();
  }
  const src = sampleCache[key];
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}
