import { NODE_CATALOG_MAP } from '../data/nodeCatalog';
import {
  ExecutionLogEntry,
  NodeExecutionResult,
  OSDElement,
  PipelineEdge,
  PipelineNode,
} from '../types/pipeline';
import { getSampleImageDataAsync, loadImageDataFromDataUrl } from './sampleImages';

declare global {
  interface Window {
    cv?: any;
  }
}

export function isOpenCVWasmReady(): boolean {
  return typeof window !== 'undefined' && Boolean(window.cv && window.cv.Mat);
}

function seededRandom(seed: number): () => number {
  let s = (seed | 0) + 1234567;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) % 100000) / 100000;
  };
}

function imageDataToDataUrl(img: ImageData): string {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

function cloneImageData(img: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
}

function resizeImageData(src: ImageData, targetW: number, targetH: number): ImageData {
  if (src.width === targetW && src.height === targetH) return src;
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = src.width;
  srcCanvas.height = src.height;
  srcCanvas.getContext('2d')!.putImageData(src, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = targetW;
  dstCanvas.height = targetH;
  const dstCtx = dstCanvas.getContext('2d')!;
  dstCtx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  return dstCtx.getImageData(0, 0, targetW, targetH);
}

function rgbToHsvPixel(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;

  let h = 0;
  if (d !== 0) {
    if (max === rf) h = ((gf - bf) / d) % 6;
    else if (max === gf) h = (bf - rf) / d + 2;
    else h = (rf - gf) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return [Math.round((h / 360) * 255), Math.round(s * 255), Math.round(v * 255)];
}

function toGrayArray(src: ImageData): Uint8Array {
  const gray = new Uint8Array(src.width * src.height);
  for (let i = 0, p = 0; i < src.data.length; i += 4, p++) {
    gray[p] = Math.round(0.299 * src.data[i] + 0.587 * src.data[i + 1] + 0.114 * src.data[i + 2]);
  }
  return gray;
}

function applyCLAHE(src: ImageData, clipLimit = 2.5, tileSize = 8): ImageData {
  if (isOpenCVWasmReady()) {
    try {
      const cv = window.cv;
      const mat = cv.matFromImageData(src);
      const lab = new cv.Mat();
      cv.cvtColor(mat, lab, cv.COLOR_RGBA2RGB);
      const lab2 = new cv.Mat();
      cv.cvtColor(lab, lab2, cv.COLOR_RGB2Lab);
      const planes = new cv.MatVector();
      cv.split(lab2, planes);
      const clahe = new cv.CLAHE(
        Math.max(0.5, clipLimit),
        new cv.Size(Math.max(2, tileSize), Math.max(2, tileSize))
      );
      const lOut = new cv.Mat();
      clahe.apply(planes.get(0), lOut);
      planes.set(0, lOut);
      cv.merge(planes, lab2);
      cv.cvtColor(lab2, lab, cv.COLOR_Lab2RGB);
      cv.cvtColor(lab, mat, cv.COLOR_RGB2RGBA);
      const out = new ImageData(new Uint8ClampedArray(mat.data), mat.cols, mat.rows);
      mat.delete();
      lab.delete();
      lab2.delete();
      planes.delete();
      clahe.delete();
      lOut.delete();
      return out;
    } catch {
      // Fallback below
    }
  }

  const { width, height, data } = src;
  const out = new Uint8ClampedArray(data.length);
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i += 4) {
    const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    hist[lum]++;
  }
  const totalPixels = width * height;
  const clipCount = Math.max(1, Math.round((clipLimit * totalPixels) / 256));
  let excess = 0;
  for (let i = 0; i < 256; i++) {
    if (hist[i] > clipCount) {
      excess += hist[i] - clipCount;
      hist[i] = clipCount;
    }
  }
  const addPerBin = Math.floor(excess / 256);
  const lut = new Uint8ClampedArray(256);
  let cdf = 0;
  for (let i = 0; i < 256; i++) {
    cdf += hist[i] + addPerBin;
    lut[i] = Math.min(255, Math.round((cdf / totalPixels) * 255));
  }
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = Math.max(1, Math.round(0.299 * r + 0.587 * g + 0.114 * b));
    const targetLum = lut[lum];
    const ratio = targetLum / lum;
    out[i] = Math.min(255, Math.round(r * ratio));
    out[i + 1] = Math.min(255, Math.round(g * ratio));
    out[i + 2] = Math.min(255, Math.round(b * ratio));
    out[i + 3] = data[i + 3];
  }
  return new ImageData(out, width, height);
}

function applyEqualizeHist(src: ImageData, luminanceOnly = true): ImageData {
  const { width, height, data } = src;
  const total = width * height;
  const out = new Uint8ClampedArray(data.length);

  if (luminanceOnly) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < data.length; i += 4) {
      const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      hist[lum]++;
    }
    const lut = new Uint8ClampedArray(256);
    let cdf = 0;
    for (let i = 0; i < 256; i++) {
      cdf += hist[i];
      lut[i] = Math.round((cdf / total) * 255);
    }
    for (let i = 0; i < data.length; i += 4) {
      const lum = Math.max(1, Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]));
      const ratio = lut[lum] / lum;
      out[i] = Math.min(255, Math.round(data[i] * ratio));
      out[i + 1] = Math.min(255, Math.round(data[i + 1] * ratio));
      out[i + 2] = Math.min(255, Math.round(data[i + 2] * ratio));
      out[i + 3] = 255;
    }
  } else {
    for (let c = 0; c < 3; c++) {
      const hist = new Uint32Array(256);
      for (let i = c; i < data.length; i += 4) hist[data[i]]++;
      const lut = new Uint8ClampedArray(256);
      let cdf = 0;
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        lut[i] = Math.round((cdf / total) * 255);
      }
      for (let i = c; i < data.length; i += 4) out[i] = lut[data[i]];
    }
    for (let i = 3; i < data.length; i += 4) out[i] = 255;
  }
  return new ImageData(out, width, height);
}

function applyToHSV(src: ImageData, previewMode: string): ImageData {
  const out = new Uint8ClampedArray(src.data.length);
  for (let i = 0; i < src.data.length; i += 4) {
    const [h, s, v] = rgbToHsvPixel(src.data[i], src.data[i + 1], src.data[i + 2]);
    if (previewMode === 'hue_only') {
      out[i] = h;
      out[i + 1] = h;
      out[i + 2] = h;
    } else if (previewMode === 'sat_only') {
      out[i] = s;
      out[i + 1] = s;
      out[i + 2] = s;
    } else {
      out[i] = h;
      out[i + 1] = s;
      out[i + 2] = v;
    }
    out[i + 3] = 255;
  }
  return new ImageData(out, src.width, src.height);
}

function applyToLAB(src: ImageData, channelView = 'composite'): ImageData {
  const out = new Uint8ClampedArray(src.data.length);
  for (let i = 0; i < src.data.length; i += 4) {
    const r = src.data[i] / 255;
    const g = src.data[i + 1] / 255;
    const b = src.data[i + 2] / 255;
    // Fast sRGB -> XYZ -> Lab approximation
    const x = 0.412453 * r + 0.35758 * g + 0.180423 * b;
    const y = 0.212671 * r + 0.71516 * g + 0.072169 * b;
    const z = 0.019334 * r + 0.119193 * g + 0.950227 * b;
    const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    const fx = f(x / 0.950456);
    const fy = f(y);
    const fz = f(z / 1.088754);
    const L = Math.min(255, Math.max(0, Math.round(((116 * fy - 16) / 100) * 255)));
    const A = Math.min(255, Math.max(0, Math.round(128 + 500 * (fx - fy) * 0.5)));
    const B = Math.min(255, Math.max(0, Math.round(128 + 200 * (fy - fz) * 0.5)));

    if (channelView === 'l_channel') {
      out[i] = out[i + 1] = out[i + 2] = L;
    } else if (channelView === 'a_channel') {
      out[i] = out[i + 1] = out[i + 2] = A;
    } else if (channelView === 'b_channel') {
      out[i] = out[i + 1] = out[i + 2] = B;
    } else {
      out[i] = L;
      out[i + 1] = A;
      out[i + 2] = B;
    }
    out[i + 3] = 255;
  }
  return new ImageData(out, src.width, src.height);
}

function sampleColorMap(t: number, palette: string): [number, number, number] {
  const clamp = (v: number) => Math.min(255, Math.max(0, Math.round(v * 255)));
  if (palette === 'jet') {
    const r = Math.min(Math.max(1.5 - Math.abs(4 * t - 3), 0), 1);
    const g = Math.min(Math.max(1.5 - Math.abs(4 * t - 2), 0), 1);
    const b = Math.min(Math.max(1.5 - Math.abs(4 * t - 1), 0), 1);
    return [clamp(r), clamp(g), clamp(b)];
  }
  if (palette === 'hot') {
    const r = Math.min(1, t * 2.5);
    const g = Math.min(1, Math.max(0, (t - 0.35) * 2.4));
    const b = Math.min(1, Math.max(0, (t - 0.72) * 3.5));
    return [clamp(r), clamp(g), clamp(b)];
  }
  if (palette === 'inferno') {
    const r = Math.min(1, Math.pow(t, 0.65) * 1.15);
    const g = Math.min(1, Math.pow(Math.max(0, t - 0.25), 1.2) * 1.4);
    const b = Math.min(1, Math.sin(t * Math.PI) * 0.7 + (t > 0.75 ? (t - 0.75) * 4 : 0));
    return [clamp(r), clamp(g), clamp(b)];
  }
  if (palette === 'viridis') {
    const r = Math.max(0, Math.min(1, -0.35 + 1.4 * t * t));
    const g = Math.max(0, Math.min(1, 0.1 + 0.85 * t));
    const b = Math.max(0, Math.min(1, 0.35 + 0.65 * Math.sin(t * 3.14)));
    return [clamp(r), clamp(g), clamp(b)];
  }
  // Turbo-inspired rainbow perceptual map
  const r = Math.max(0, Math.min(1, 0.13 + Math.sin((t - 0.28) * 4.2) * 0.85));
  const g = Math.max(0, Math.min(1, Math.sin(t * Math.PI) * 0.98));
  const b = Math.max(0, Math.min(1, 0.22 + Math.cos(t * 3.7) * 0.75));
  return [clamp(r), clamp(g), clamp(b)];
}

function applyColorMap(src: ImageData, palette = 'turbo'): ImageData {
  const gray = toGrayArray(src);
  const out = new Uint8ClampedArray(src.data.length);
  for (let p = 0, i = 0; p < gray.length; p++, i += 4) {
    const [r, g, b] = sampleColorMap(gray[p] / 255, palette);
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = b;
    out[i + 3] = 255;
  }
  return new ImageData(out, src.width, src.height);
}

function applyInRange(src: ImageData, params: Record<string, unknown>): ImageData {
  const lH = Number(params.lowerH ?? 0);
  const uH = Number(params.upperH ?? 180);
  const lS = Number(params.lowerS ?? 40);
  const uS = Number(params.upperS ?? 255);
  const lV = Number(params.lowerV ?? 40);
  const uV = Number(params.upperV ?? 255);

  const out = new Uint8ClampedArray(src.data.length);
  for (let i = 0; i < src.data.length; i += 4) {
    const c1 = src.data[i];
    const c2 = src.data[i + 1];
    const c3 = src.data[i + 2];
    const match = c1 >= lH && c1 <= uH && c2 >= lS && c2 <= uS && c3 >= lV && c3 <= uV;
    const val = match ? 255 : 0;
    out[i] = val;
    out[i + 1] = val;
    out[i + 2] = val;
    out[i + 3] = 255;
  }
  return new ImageData(out, src.width, src.height);
}

function applyGaussianBlur(src: ImageData, kernelSize = 5): ImageData {
  const k = Math.max(1, Math.round(kernelSize) | 1);
  if (k <= 1) return cloneImageData(src);
  const canvas = document.createElement('canvas');
  canvas.width = src.width;
  canvas.height = src.height;
  const ctx = canvas.getContext('2d')!;
  const tmp = document.createElement('canvas');
  tmp.width = src.width;
  tmp.height = src.height;
  tmp.getContext('2d')!.putImageData(src, 0, 0);
  ctx.filter = `blur(${Math.max(0.5, (k - 1) * 0.45)}px)`;
  ctx.drawImage(tmp, 0, 0);
  return ctx.getImageData(0, 0, src.width, src.height);
}

function applyMedianBlur(src: ImageData, kernelSize = 5): ImageData {
  const { width, height, data } = src;
  const r = Math.max(1, Math.min(4, Math.floor(kernelSize / 2)));
  const out = new Uint8ClampedArray(data.length);
  const winR: number[] = [];
  const winG: number[] = [];
  const winB: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      winR.length = 0;
      winG.length = 0;
      winB.length = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = Math.min(height - 1, Math.max(0, y + dy));
        for (let dx = -r; dx <= r; dx++) {
          const nx = Math.min(width - 1, Math.max(0, x + dx));
          const idx = (ny * width + nx) * 4;
          winR.push(data[idx]);
          winG.push(data[idx + 1]);
          winB.push(data[idx + 2]);
        }
      }
      winR.sort((a, b) => a - b);
      winG.sort((a, b) => a - b);
      winB.sort((a, b) => a - b);
      const mid = winR.length >> 1;
      const oIdx = (y * width + x) * 4;
      out[oIdx] = winR[mid];
      out[oIdx + 1] = winG[mid];
      out[oIdx + 2] = winB[mid];
      out[oIdx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyBilateralFilter(src: ImageData, diameter = 7, sigmaColor = 55, sigmaSpace = 35): ImageData {
  const { width, height, data } = src;
  const r = Math.max(1, Math.min(5, Math.floor(diameter / 2)));
  const out = new Uint8ClampedArray(data.length);
  const twoSigmaC2 = 2 * sigmaColor * sigmaColor;
  const twoSigmaS2 = 2 * sigmaSpace * sigmaSpace;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const cIdx = (y * width + x) * 4;
      const cr = data[cIdx];
      const cg = data[cIdx + 1];
      const cb = data[cIdx + 2];
      let sumW = 0;
      let sumR = 0;
      let sumG = 0;
      let sumB = 0;

      for (let dy = -r; dy <= r; dy++) {
        const ny = Math.min(height - 1, Math.max(0, y + dy));
        for (let dx = -r; dx <= r; dx++) {
          const nx = Math.min(width - 1, Math.max(0, x + dx));
          const nIdx = (ny * width + nx) * 4;
          const dr = data[nIdx] - cr;
          const dg = data[nIdx + 1] - cg;
          const db = data[nIdx + 2] - cb;
          const colorDist2 = dr * dr + dg * dg + db * db;
          const spaceDist2 = dx * dx + dy * dy;
          const w = Math.exp(-colorDist2 / twoSigmaC2 - spaceDist2 / twoSigmaS2);
          sumW += w;
          sumR += data[nIdx] * w;
          sumG += data[nIdx + 1] * w;
          sumB += data[nIdx + 2] * w;
        }
      }
      out[cIdx] = Math.round(sumR / sumW);
      out[cIdx + 1] = Math.round(sumG / sumW);
      out[cIdx + 2] = Math.round(sumB / sumW);
      out[cIdx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyBoxBlur(src: ImageData, kernelSize = 5): ImageData {
  const { width, height, data } = src;
  const r = Math.max(1, Math.min(10, Math.floor(kernelSize / 2)));
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let cnt = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = Math.min(height - 1, Math.max(0, y + dy));
        for (let dx = -r; dx <= r; dx++) {
          const nx = Math.min(width - 1, Math.max(0, x + dx));
          const idx = (ny * width + nx) * 4;
          sr += data[idx];
          sg += data[idx + 1];
          sb += data[idx + 2];
          cnt++;
        }
      }
      const oIdx = (y * width + x) * 4;
      out[oIdx] = Math.round(sr / cnt);
      out[oIdx + 1] = Math.round(sg / cnt);
      out[oIdx + 2] = Math.round(sb / cnt);
      out[oIdx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applySobel(src: ImageData, axis = 'xy', gain = 1.0): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx =
        -gray[(y - 1) * width + (x - 1)] +
        gray[(y - 1) * width + (x + 1)] -
        2 * gray[y * width + (x - 1)] +
        2 * gray[y * width + (x + 1)] -
        gray[(y + 1) * width + (x - 1)] +
        gray[(y + 1) * width + (x + 1)];
      const gy =
        -gray[(y - 1) * width + (x - 1)] -
        2 * gray[(y - 1) * width + x] -
        gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] +
        2 * gray[(y + 1) * width + x] +
        gray[(y + 1) * width + (x + 1)];
      const raw = axis === 'x' ? Math.abs(gx) : axis === 'y' ? Math.abs(gy) : Math.hypot(gx, gy);
      const val = Math.min(255, Math.round(raw * gain));
      const idx = (y * width + x) * 4;
      out[idx] = out[idx + 1] = out[idx + 2] = val;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyLaplacian(src: ImageData, gain = 1.5): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const lap =
        gray[(y - 1) * width + x] +
        gray[(y + 1) * width + x] +
        gray[y * width + (x - 1)] +
        gray[y * width + (x + 1)] -
        4 * gray[y * width + x];
      const val = Math.min(255, Math.round(Math.abs(lap) * gain));
      const idx = (y * width + x) * 4;
      out[idx] = out[idx + 1] = out[idx + 2] = val;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyAdaptiveThreshold(src: ImageData, blockSize = 11, cConst = 4): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const r = Math.max(1, Math.floor(Math.max(3, blockSize) / 2));

  const integral = new Int32Array((width + 1) * (height + 1));
  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      rowSum += gray[(y - 1) * width + (x - 1)];
      integral[y * (width + 1) + x] = integral[(y - 1) * (width + 1) + x] + rowSum;
    }
  }

  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(height - 1, y + r);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(width - 1, x + r);
      const count = (x1 - x0 + 1) * (y1 - y0 + 1);
      const sum =
        integral[(y1 + 1) * (width + 1) + (x1 + 1)] -
        integral[y0 * (width + 1) + (x1 + 1)] -
        integral[(y1 + 1) * (width + 1) + x0] +
        integral[y0 * (width + 1) + x0];
      const thresh = sum / count - cConst;
      const val = gray[y * width + x] > thresh ? 255 : 0;
      const idx = (y * width + x) * 4;
      out[idx] = val;
      out[idx + 1] = val;
      out[idx + 2] = val;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyOtsuThreshold(src: ImageData, invert = false): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];

  let sumB = 0;
  let wB = 0;
  let maxVar = 0;
  let thresh = 128;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const varBetween = wB * wF * (mB - mF) * (mB - mF);
    if (varBetween > maxVar) {
      maxVar = varBetween;
      thresh = t;
    }
  }

  const out = new Uint8ClampedArray(width * height * 4);
  for (let p = 0, i = 0; p < gray.length; p++, i += 4) {
    const fg = gray[p] > thresh;
    const val = (invert ? !fg : fg) ? 255 : 0;
    out[i] = out[i + 1] = out[i + 2] = val;
    out[i + 3] = 255;
  }
  return new ImageData(out, width, height);
}

function applyFindContours(
  maskImg: ImageData,
  overlayImg: ImageData | undefined,
  minArea = 40,
  drawBoxes = true,
  color = '#22C55E'
): ImageData {
  const { width, height } = maskImg;
  const gray = toGrayArray(maskImg);
  const visited = new Uint8Array(width * height);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  if (overlayImg) {
    const resized = resizeImageData(overlayImg, width, height);
    ctx.putImageData(resized, 0, 0);
  } else {
    ctx.fillStyle = '#090B0E';
    ctx.fillRect(0, 0, width, height);
  }

  const boxes: Array<{ minX: number; minY: number; maxX: number; maxY: number; area: number }> = [];
  const boundaryMask = new Uint8Array(width * height);
  const stack = new Int32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const start = y * width + x;
      if (gray[start] <= 127 || visited[start]) continue;

      let sp = 0;
      stack[sp++] = start;
      visited[start] = 1;
      let area = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      const compPixels: number[] = [];

      while (sp > 0) {
        const cur = stack[--sp];
        const cx = cur % width;
        const cy = (cur / width) | 0;
        area++;
        compPixels.push(cur);
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const nbs = [cur - 1, cur + 1, cur - width, cur + width];
        for (const nb of nbs) {
          if (nb >= 0 && nb < width * height && !visited[nb] && gray[nb] > 127) {
            visited[nb] = 1;
            stack[sp++] = nb;
          }
        }
      }

      if (area >= minArea) {
        boxes.push({ minX, minY, maxX, maxY, area });
        for (const p of compPixels) {
          const px = p % width;
          const py = (p / width) | 0;
          if (
            px === 0 ||
            py === 0 ||
            px === width - 1 ||
            py === height - 1 ||
            gray[p - 1] <= 127 ||
            gray[p + 1] <= 127 ||
            gray[p - width] <= 127 ||
            gray[p + width] <= 127
          ) {
            boundaryMask[p] = 1;
          }
        }
      }
    }
  }

  const outImg = ctx.getImageData(0, 0, width, height);
  const [cr, cg, cb] = hexToRgb(color);
  for (let p = 0, i = 0; p < boundaryMask.length; p++, i += 4) {
    if (boundaryMask[p]) {
      outImg.data[i] = cr;
      outImg.data[i + 1] = cg;
      outImg.data[i + 2] = cb;
      outImg.data[i + 3] = 255;
    }
  }
  ctx.putImageData(outImg, 0, 0);

  if (drawBoxes) {
    ctx.strokeStyle = '#F58220';
    ctx.lineWidth = 1.5;
    boxes.slice(0, 40).forEach((b) => {
      ctx.strokeRect(b.minX, b.minY, Math.max(2, b.maxX - b.minX), Math.max(2, b.maxY - b.minY));
    });
  }

  return ctx.getImageData(0, 0, width, height);
}

function applyCornerDetect(src: ImageData, maxCorners = 45, threshold = 35, color = '#F58220'): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const corners: Array<{ x: number; y: number; score: number }> = [];

  for (let y = 4; y < height - 4; y += 3) {
    for (let x = 4; x < width - 4; x += 3) {
      let ixx = 0;
      let iyy = 0;
      let ixy = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const p = (y + dy) * width + (x + dx);
          const gx = gray[p + 1] - gray[p - 1];
          const gy = gray[p + width] - gray[p - width];
          ixx += gx * gx;
          iyy += gy * gy;
          ixy += gx * gy;
        }
      }
      const det = ixx * iyy - ixy * ixy;
      const trace = ixx + iyy;
      const r = (det - 0.04 * trace * trace) / 1e6;
      if (r > threshold) {
        corners.push({ x, y, score: r });
      }
    }
  }

  corners.sort((a, b) => b.score - a.score);
  const chosen: Array<{ x: number; y: number }> = [];
  for (const c of corners) {
    if (chosen.every((k) => Math.hypot(k.x - c.x, k.y - c.y) > 10)) {
      chosen.push({ x: c.x, y: c.y });
      if (chosen.length >= maxCorners) break;
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(src, 0, 0);
  ctx.fillStyle = color;
  ctx.strokeStyle = '#0B0D10';
  ctx.lineWidth = 1;
  chosen.forEach((pt) => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  return ctx.getImageData(0, 0, width, height);
}

function applyCanny(src: ImageData, t1 = 60, t2 = 150): ImageData {
  const { width, height } = src;
  const gray = toGrayArray(src);
  const mag = new Float32Array(width * height);
  const dir = new Uint8Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const gx =
        -gray[(y - 1) * width + (x - 1)] +
        gray[(y - 1) * width + (x + 1)] -
        2 * gray[y * width + (x - 1)] +
        2 * gray[y * width + (x + 1)] -
        gray[(y + 1) * width + (x - 1)] +
        gray[(y + 1) * width + (x + 1)];
      const gy =
        -gray[(y - 1) * width + (x - 1)] -
        2 * gray[(y - 1) * width + x] -
        gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] +
        2 * gray[(y + 1) * width + x] +
        gray[(y + 1) * width + (x + 1)];
      mag[p] = Math.hypot(gx, gy);
      const angle = (Math.atan2(gy, gx) * 180) / Math.PI;
      const a = angle < 0 ? angle + 180 : angle;
      if (a < 22.5 || a >= 157.5) dir[p] = 0;
      else if (a < 67.5) dir[p] = 1;
      else if (a < 112.5) dir[p] = 2;
      else dir[p] = 3;
    }
  }

  const out = new Uint8ClampedArray(width * height * 4);
  const low = Math.min(t1, t2);
  const high = Math.max(t1, t2);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const m = mag[p];
      let n1 = 0;
      let n2 = 0;
      if (dir[p] === 0) {
        n1 = mag[p - 1];
        n2 = mag[p + 1];
      } else if (dir[p] === 1) {
        n1 = mag[(y - 1) * width + x + 1];
        n2 = mag[(y + 1) * width + x - 1];
      } else if (dir[p] === 2) {
        n1 = mag[(y - 1) * width + x];
        n2 = mag[(y + 1) * width + x];
      } else {
        n1 = mag[(y - 1) * width + x - 1];
        n2 = mag[(y + 1) * width + x + 1];
      }
      let edge = 0;
      if (m >= n1 && m >= n2) {
        if (m >= high) edge = 255;
        else if (m >= low) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (mag[(y + dy) * width + (x + dx)] >= high) {
                edge = 255;
                break;
              }
            }
          }
        }
      }
      const idx = p * 4;
      out[idx] = edge;
      out[idx + 1] = edge;
      out[idx + 2] = edge;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, width, height);
}

function applyMorphology(src: ImageData, mode: 'dilate' | 'erode', kernelSize = 3, iterations = 1): ImageData {
  const { width, height } = src;
  const r = Math.max(1, Math.floor(Math.max(1, kernelSize) / 2));
  let current = new Uint8ClampedArray(src.data);

  for (let iter = 0; iter < Math.max(1, Math.min(6, iterations)); iter++) {
    const next = new Uint8ClampedArray(current.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let bestR = mode === 'dilate' ? 0 : 255;
        let bestG = mode === 'dilate' ? 0 : 255;
        let bestB = mode === 'dilate' ? 0 : 255;
        for (let dy = -r; dy <= r; dy++) {
          const ny = Math.min(height - 1, Math.max(0, y + dy));
          for (let dx = -r; dx <= r; dx++) {
            const nx = Math.min(width - 1, Math.max(0, x + dx));
            const idx = (ny * width + nx) * 4;
            if (mode === 'dilate') {
              if (current[idx] > bestR) bestR = current[idx];
              if (current[idx + 1] > bestG) bestG = current[idx + 1];
              if (current[idx + 2] > bestB) bestB = current[idx + 2];
            } else {
              if (current[idx] < bestR) bestR = current[idx];
              if (current[idx + 1] < bestG) bestG = current[idx + 1];
              if (current[idx + 2] < bestB) bestB = current[idx + 2];
            }
          }
        }
        const outIdx = (y * width + x) * 4;
        next[outIdx] = bestR;
        next[outIdx + 1] = bestG;
        next[outIdx + 2] = bestB;
        next[outIdx + 3] = 255;
      }
    }
    current = next;
  }
  return new ImageData(current, width, height);
}

function applyDrawHistogram(src: ImageData, mode = 'channels', plotW = 512, plotH = 256): ImageData {
  const w = Math.max(128, Math.min(512, plotW));
  const h = Math.max(128, Math.min(512, plotH));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, w, h);

  const rHist = new Uint32Array(256);
  const gHist = new Uint32Array(256);
  const bHist = new Uint32Array(256);

  for (let i = 0; i < src.data.length; i += 4) {
    if (mode === 'luminance') {
      const lum = Math.round(0.299 * src.data[i] + 0.587 * src.data[i + 1] + 0.114 * src.data[i + 2]);
      rHist[lum]++;
    } else if (mode === 'hsv') {
      const [hv, sv, vv] = rgbToHsvPixel(src.data[i], src.data[i + 1], src.data[i + 2]);
      rHist[hv]++;
      gHist[sv]++;
      bHist[vv]++;
    } else {
      rHist[src.data[i]]++;
      gHist[src.data[i + 1]]++;
      bHist[src.data[i + 2]]++;
    }
  }

  let maxVal = 1;
  for (let i = 2; i < 254; i++) {
    maxVal = Math.max(maxVal, rHist[i], gHist[i], bHist[i]);
  }

  const drawChannel = (hist: Uint32Array, fill: string, stroke: string) => {
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let bin = 0; bin < 256; bin++) {
      const x = (bin / 255) * w;
      const norm = Math.min(1, hist[bin] / maxVal);
      const y = h - norm * (h * 0.88);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  if (mode === 'luminance') {
    drawChannel(rHist, 'rgba(226, 232, 240, 0.72)', 'rgba(248, 250, 252, 0.95)');
  } else {
    drawChannel(bHist, 'rgba(59, 130, 246, 0.65)', 'rgba(96, 165, 250, 0.95)');
    drawChannel(gHist, 'rgba(34, 197, 94, 0.62)', 'rgba(74, 222, 128, 0.95)');
    drawChannel(rHist, 'rgba(239, 68, 68, 0.62)', 'rgba(248, 113, 113, 0.95)');
  }

  return ctx.getImageData(0, 0, w, h);
}

function applyArithmetic(
  imgA: ImageData,
  imgB: ImageData | undefined,
  mask: ImageData | undefined,
  operation = 'add',
  alpha = 0.65
): ImageData {
  if (!imgB) return cloneImageData(imgA);
  const bResized = resizeImageData(imgB, imgA.width, imgA.height);
  const mResized = mask ? resizeImageData(mask, imgA.width, imgA.height) : undefined;
  const out = new Uint8ClampedArray(imgA.data.length);

  for (let i = 0; i < imgA.data.length; i += 4) {
    const maskActive = !mResized || mResized.data[i] > 127;
    if (!maskActive) {
      out[i] = imgA.data[i];
      out[i + 1] = imgA.data[i + 1];
      out[i + 2] = imgA.data[i + 2];
      out[i + 3] = 255;
      continue;
    }

    for (let c = 0; c < 3; c++) {
      const a = imgA.data[i + c];
      const b = bResized.data[i + c];
      let v = a;
      if (operation === 'add') v = Math.min(255, a + b);
      else if (operation === 'subtract') v = Math.max(0, a - b);
      else if (operation === 'absdiff') v = Math.abs(a - b);
      else if (operation === 'bitwise_or') v = a | b;
      else if (operation === 'bitwise_and') v = a & b;
      else if (operation === 'bitwise_xor') v = a ^ b;
      else if (operation === 'blend') v = Math.round(a * alpha + b * (1 - alpha));
      out[i + c] = v;
    }
    out[i + 3] = 255;
  }
  return new ImageData(out, imgA.width, imgA.height);
}

function applyMaskOp(img: ImageData, mask: ImageData | undefined, invertMask = false): ImageData {
  if (!mask) return cloneImageData(img);
  const mResized = resizeImageData(mask, img.width, img.height);
  const out = new Uint8ClampedArray(img.data.length);
  for (let i = 0; i < img.data.length; i += 4) {
    const mVal = mResized.data[i] > 127;
    const keep = invertMask ? !mVal : mVal;
    out[i] = keep ? img.data[i] : 0;
    out[i + 1] = keep ? img.data[i + 1] : 0;
    out[i + 2] = keep ? img.data[i + 2] : 0;
    out[i + 3] = 255;
  }
  return new ImageData(out, img.width, img.height);
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  const num = parseInt(clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean, 16);
  if (Number.isNaN(num)) return [34, 197, 94];
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function sampleBilinearBorder(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  sx: number,
  sy: number,
  borderMode: string
): [number, number, number] {
  let x = sx;
  let y = sy;
  if (x < 0 || x >= w - 1 || y < 0 || y >= h - 1) {
    if (borderMode === 'constant') {
      if (x < -0.5 || x > w - 0.5 || y < -0.5 || y > h - 0.5) return [0, 0, 0];
    } else if (borderMode === 'reflect') {
      while (x < 0 || x >= w) x = x < 0 ? -x : 2 * w - 2 - x;
      while (y < 0 || y >= h) y = y < 0 ? -y : 2 * h - 2 - y;
    }
    x = Math.min(w - 1, Math.max(0, x));
    y = Math.min(h - 1, Math.max(0, y));
  }

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;

  const i00 = (y0 * w + x0) * 4;
  const i10 = (y0 * w + x1) * 4;
  const i01 = (y1 * w + x0) * 4;
  const i11 = (y1 * w + x1) * 4;

  const interp = (off: number) =>
    Math.round(
      data[i00 + off] * (1 - fx) * (1 - fy) +
        data[i10 + off] * fx * (1 - fy) +
        data[i01 + off] * (1 - fx) * fy +
        data[i11 + off] * fx * fy
    );

  return [interp(0), interp(1), interp(2)];
}

function applyWarpAffine(
  src: ImageData,
  angleDeg = 18,
  scale = 0.92,
  tx = 0,
  ty = 0,
  shearX = 0,
  borderMode = 'constant'
): ImageData {
  const { width: w, height: h, data } = src;
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad) * scale;
  const sin = Math.sin(rad) * scale;
  const cx = w / 2;
  const cy = h / 2;

  // Forward affine M: [a b c; d e f] around center + shear + translate
  const a = cos;
  const b = sin + shearX * cos;
  const d = -sin;
  const e = cos - shearX * sin;
  const c = cx - a * cx - b * cy + tx;
  const f = cy - d * cx - e * cy + ty;

  if (isOpenCVWasmReady()) {
    try {
      const cv = window.cv;
      const srcMat = cv.matFromImageData(src);
      const dstMat = new cv.Mat();
      const M = cv.matFromArray(2, 3, cv.CV_64FC1, [a, b, c, d, e, f]);
      const bFlag =
        borderMode === 'reflect'
          ? cv.BORDER_REFLECT
          : borderMode === 'replicate'
            ? cv.BORDER_REPLICATE
            : cv.BORDER_CONSTANT;
      cv.warpAffine(srcMat, dstMat, M, new cv.Size(w, h), cv.INTER_LINEAR, bFlag, new cv.Scalar(0, 0, 0, 255));
      const out = new ImageData(new Uint8ClampedArray(dstMat.data), dstMat.cols, dstMat.rows);
      srcMat.delete();
      dstMat.delete();
      M.delete();
      return out;
    } catch {
      // Fallback to inverse bilinear sampling below
    }
  }

  const det = a * e - b * d;
  if (Math.abs(det) < 1e-6) return cloneImageData(src);
  const invA = e / det;
  const invB = -b / det;
  const invD = -d / det;
  const invE = a / det;

  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - c;
      const dy = y - f;
      const sx = invA * dx + invB * dy;
      const sy = invD * dx + invE * dy;
      const [r, g, bl] = sampleBilinearBorder(data, w, h, sx, sy, borderMode);
      const idx = (y * w + x) * 4;
      out[idx] = r;
      out[idx + 1] = g;
      out[idx + 2] = bl;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, w, h);
}

function solveLinear8(A: number[][], b: number[]): number[] {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > Math.abs(M[maxRow][i])) maxRow = r;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    const pivot = Math.abs(M[i][i]) < 1e-10 ? 1e-10 : M[i][i];
    for (let j = i; j <= n; j++) M[i][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = M[r][i];
      for (let j = i; j <= n; j++) M[r][j] -= factor * M[i][j];
    }
  }
  return M.map((row) => row[n]);
}

function computeHomography(
  srcPts: Array<[number, number]>,
  dstPts: Array<[number, number]>
): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = srcPts[i];
    const [u, v] = dstPts[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solveLinear8(A, b);
  return [...h, 1];
}

function applyWarpPerspective(
  src: ImageData,
  params: Record<string, unknown>
): ImageData {
  const { width: w, height: h, data } = src;
  const tlX = (Number(params.tlX ?? 12) / 100) * w;
  const tlY = (Number(params.tlY ?? 8) / 100) * h;
  const trX = (Number(params.trX ?? 88) / 100) * w;
  const trY = (Number(params.trY ?? 12) / 100) * h;
  const brX = (Number(params.brX ?? 98) / 100) * w;
  const brY = (Number(params.brY ?? 94) / 100) * h;
  const blX = (Number(params.blX ?? 2) / 100) * w;
  const blY = (Number(params.blY ?? 92) / 100) * h;
  const borderMode = String(params.borderMode || 'constant');

  const srcPts: Array<[number, number]> = [
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1],
  ];
  const dstPts: Array<[number, number]> = [
    [tlX, tlY],
    [trX, trY],
    [brX, brY],
    [blX, blY],
  ];

  if (isOpenCVWasmReady()) {
    try {
      const cv = window.cv;
      const srcMat = cv.matFromImageData(src);
      const dstMat = new cv.Mat();
      const sMat = cv.matFromArray(4, 1, cv.CV_32FC2, srcPts.flat());
      const dMat = cv.matFromArray(4, 1, cv.CV_32FC2, dstPts.flat());
      const M = cv.getPerspectiveTransform(sMat, dMat);
      const bFlag = borderMode === 'replicate' ? cv.BORDER_REPLICATE : cv.BORDER_CONSTANT;
      cv.warpPerspective(srcMat, dstMat, M, new cv.Size(w, h), cv.INTER_LINEAR, bFlag, new cv.Scalar(0, 0, 0, 255));
      const out = new ImageData(new Uint8ClampedArray(dstMat.data), dstMat.cols, dstMat.rows);
      srcMat.delete();
      dstMat.delete();
      sMat.delete();
      dMat.delete();
      M.delete();
      return out;
    } catch {
      // Fallback to projective inverse DLT homography below
    }
  }

  // Inverse homography maps dst (x, y) -> src (sx, sy)
  const invH = computeHomography(dstPts, srcPts);
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const denom = invH[6] * x + invH[7] * y + invH[8];
      const sx = (invH[0] * x + invH[1] * y + invH[2]) / (Math.abs(denom) < 1e-6 ? 1e-6 : denom);
      const sy = (invH[3] * x + invH[4] * y + invH[5]) / (Math.abs(denom) < 1e-6 ? 1e-6 : denom);
      const [r, g, bl] = sampleBilinearBorder(data, w, h, sx, sy, borderMode);
      const idx = (y * w + x) * 4;
      out[idx] = r;
      out[idx + 1] = g;
      out[idx + 2] = bl;
      out[idx + 3] = 255;
    }
  }
  return new ImageData(out, w, h);
}

export function renderOSDElementsToImageData(
  baseImg: ImageData,
  elements: OSDElement[],
  globalOpacity = 1.0,
  showGridBurnIn = false
): ImageData {
  const w = baseImg.width;
  const h = baseImg.height;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(baseImg, 0, 0);

  if (showGridBurnIn) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1.25;
    const m = 12;
    const len = 18;
    ctx.beginPath();
    ctx.moveTo(m, m + len);
    ctx.lineTo(m, m);
    ctx.lineTo(m + len, m);
    ctx.moveTo(w - m - len, m);
    ctx.lineTo(w - m, m);
    ctx.lineTo(w - m, m + len);
    ctx.moveTo(m, h - m - len);
    ctx.lineTo(m, h - m);
    ctx.lineTo(m + len, h - m);
    ctx.moveTo(w - m - len, h - m);
    ctx.lineTo(w - m, h - m);
    ctx.lineTo(w - m, h - m - len);
    ctx.stroke();
    ctx.restore();
  }

  if (isOpenCVWasmReady()) {
    try {
      const cv = window.cv;
      const baseMat = cv.matFromImageData(ctx.getImageData(0, 0, w, h));
      for (const el of elements) {
        if (!el.visible) continue;
        const [r, g, b] = hexToRgb(el.color || '#22C55E');
        const col = new cv.Scalar(r, g, b, Math.round(255 * (el.opacity ?? 1) * globalOpacity));
        const thickness = el.filled ? -1 : Math.max(1, Math.round(el.thickness || 2));
        const px = Math.round(el.x * w);
        const py = Math.round(el.y * h);
        const pw = Math.round(el.width * w);
        const ph = Math.round(el.height * h);

        if (el.kind === 'rect') {
          cv.rectangle(baseMat, new cv.Point(px, py), new cv.Point(px + pw, py + ph), col, thickness, cv.LINE_AA);
        } else if (el.kind === 'bbox_label') {
          cv.rectangle(baseMat, new cv.Point(px, py), new cv.Point(px + pw, py + ph), col, Math.max(1, Math.round(el.thickness || 2)), cv.LINE_AA);
          const label = el.text || 'OBJECT';
          const fScale = el.fontScale ?? 0.5;
          const tagH = Math.max(18, Math.round(22 * fScale * 1.4));
          cv.rectangle(baseMat, new cv.Point(px, Math.max(0, py - tagH)), new cv.Point(px + Math.min(pw, label.length * 9 + 14), py), col, -1);
          cv.putText(
            baseMat,
            label,
            new cv.Point(px + 5, Math.max(12, py - 5)),
            cv.FONT_HERSHEY_SIMPLEX,
            fScale,
            new cv.Scalar(11, 13, 16, 255),
            1,
            cv.LINE_AA
          );
        } else if (el.kind === 'circle') {
          const rad = Math.max(4, Math.round(Math.min(pw, ph) / 2));
          cv.circle(baseMat, new cv.Point(px + Math.round(pw / 2), py + Math.round(ph / 2)), rad, col, thickness, cv.LINE_AA);
        } else if (el.kind === 'line') {
          const x2 = Math.round((el.x2 ?? el.x + el.width) * w);
          const y2 = Math.round((el.y2 ?? el.y + el.height) * h);
          cv.line(baseMat, new cv.Point(px, py), new cv.Point(x2, y2), col, Math.max(1, Math.round(el.thickness || 2)), cv.LINE_AA);
        } else if (el.kind === 'arrow') {
          const x2 = Math.round((el.x2 ?? el.x + el.width) * w);
          const y2 = Math.round((el.y2 ?? el.y + el.height) * h);
          cv.arrowedLine(baseMat, new cv.Point(px, py), new cv.Point(x2, y2), col, Math.max(1, Math.round(el.thickness || 2)), cv.LINE_AA, 0, 0.22);
        } else if (el.kind === 'text') {
          cv.putText(
            baseMat,
            el.text || 'OSD TEXT',
            new cv.Point(px, py + Math.round(14 * (el.fontScale ?? 0.55) * 1.5)),
            cv.FONT_HERSHEY_SIMPLEX,
            el.fontScale ?? 0.55,
            col,
            Math.max(1, Math.round(el.thickness || 1.5)),
            cv.LINE_AA
          );
        } else if (el.kind === 'reticle') {
          const rad = Math.max(8, Math.round(Math.min(pw, ph) / 2));
          cv.circle(baseMat, new cv.Point(px, py), rad, col, Math.max(1, Math.round(el.thickness || 1.5)), cv.LINE_AA);
          cv.line(baseMat, new cv.Point(px - rad - 8, py), new cv.Point(px - 4, py), col, 1, cv.LINE_AA);
          cv.line(baseMat, new cv.Point(px + 4, py), new cv.Point(px + rad + 8, py), col, 1, cv.LINE_AA);
          cv.line(baseMat, new cv.Point(px, py - rad - 8), new cv.Point(px, py - 4), col, 1, cv.LINE_AA);
          cv.line(baseMat, new cv.Point(px, py + 4), new cv.Point(px, py + rad + 8), col, 1, cv.LINE_AA);
        }
      }
      const out = new ImageData(new Uint8ClampedArray(baseMat.data), baseMat.cols, baseMat.rows);
      baseMat.delete();
      return out;
    } catch {
      // Fallback below
    }
  }

  for (const el of elements) {
    if (!el.visible) continue;
    ctx.save();
    ctx.globalAlpha = Math.max(0.05, Math.min(1, (el.opacity ?? 1) * globalOpacity));
    ctx.strokeStyle = el.color || '#22C55E';
    ctx.fillStyle = el.color || '#22C55E';
    ctx.lineWidth = Math.max(1, el.thickness || 2);

    const px = el.x * w;
    const py = el.y * h;
    const pw = el.width * w;
    const ph = el.height * h;

    if (el.kind === 'rect') {
      if (el.filled) {
        ctx.globalAlpha *= 0.35;
        ctx.fillRect(px, py, pw, ph);
        ctx.globalAlpha = Math.max(0.05, Math.min(1, (el.opacity ?? 1) * globalOpacity));
      }
      ctx.strokeRect(px, py, pw, ph);
    } else if (el.kind === 'bbox_label') {
      ctx.strokeRect(px, py, pw, ph);
      const label = el.text || 'OBJECT';
      const fontSize = Math.max(10, Math.round(20 * (el.fontScale ?? 0.52)));
      ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
      const textWidth = ctx.measureText(label).width;
      const tagW = textWidth + 10;
      const tagH = fontSize + 6;
      const tagY = py - tagH >= 0 ? py - tagH : py;
      ctx.fillRect(px, tagY, tagW, tagH);
      ctx.fillStyle = '#0B0D10';
      ctx.fillText(label, px + 5, tagY + fontSize);
    } else if (el.kind === 'circle') {
      const cx = px + pw / 2;
      const cy = py + ph / 2;
      const rad = Math.max(4, Math.min(pw, ph) / 2);
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      if (el.filled) {
        ctx.globalAlpha *= 0.35;
        ctx.fill();
        ctx.globalAlpha = Math.max(0.05, Math.min(1, (el.opacity ?? 1) * globalOpacity));
      }
      ctx.stroke();
    } else if (el.kind === 'line') {
      const x2 = (el.x2 ?? el.x + el.width) * w;
      const y2 = (el.y2 ?? el.y + el.height) * h;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    } else if (el.kind === 'arrow') {
      const x2 = (el.x2 ?? el.x + el.width) * w;
      const y2 = (el.y2 ?? el.y + el.height) * h;
      const angle = Math.atan2(y2 - py, x2 - px);
      const headLen = Math.max(8, (el.thickness || 2) * 5);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
      ctx.stroke();
    } else if (el.kind === 'text') {
      const fontSize = Math.max(11, Math.round(24 * (el.fontScale ?? 0.5)));
      ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
      const txt = el.text || 'OSD TEXT';
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = 'rgba(11, 13, 16, 0.58)';
      ctx.fillRect(px - 4, py - 2, tw + 8, fontSize + 7);
      ctx.fillStyle = el.color || '#F8FAFC';
      ctx.fillText(txt, px, py + fontSize);
    } else if (el.kind === 'reticle') {
      const rad = Math.max(10, Math.min(pw, ph) / 2);
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - rad - 10, py);
      ctx.lineTo(px - 5, py);
      ctx.moveTo(px + 5, py);
      ctx.lineTo(px + rad + 10, py);
      ctx.moveTo(px, py - rad - 10);
      ctx.lineTo(px, py - 5);
      ctx.moveTo(px, py + 5);
      ctx.lineTo(px, py + rad + 10);
      ctx.stroke();
    }

    ctx.restore();
  }

  return ctx.getImageData(0, 0, w, h);
}

function getTopologicalOrder(
  nodes: PipelineNode[],
  edges: PipelineEdge[],
  stopAtNodeId?: string
): PipelineNode[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const adj = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  nodes.forEach((n) => {
    adj.set(n.id, []);
    inDegree.set(n.id, 0);
  });

  edges.forEach((e) => {
    if (nodeMap.has(e.source) && nodeMap.has(e.target)) {
      adj.get(e.source)!.push(e.target);
      inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
    }
  });

  let allowedIds: Set<string> | null = null;
  if (stopAtNodeId && nodeMap.has(stopAtNodeId)) {
    allowedIds = new Set<string>([stopAtNodeId]);
    const rev = new Map<string, string[]>();
    nodes.forEach((n) => rev.set(n.id, []));
    edges.forEach((e) => {
      if (rev.has(e.target)) rev.get(e.target)!.push(e.source);
    });
    const queue = [stopAtNodeId];
    while (queue.length > 0) {
      const cur = queue.shift()!;
      for (const parent of rev.get(cur) || []) {
        if (!allowedIds.has(parent)) {
          allowedIds.add(parent);
          queue.push(parent);
        }
      }
    }
  }

  const queue: string[] = [];
  inDegree.forEach((deg, id) => {
    if (deg === 0 && (!allowedIds || allowedIds.has(id))) queue.push(id);
  });

  const sorted: PipelineNode[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    const node = nodeMap.get(id);
    if (node) sorted.push(node);
    for (const nextId of adj.get(id) || []) {
      const nextDeg = (inDegree.get(nextId) ?? 1) - 1;
      inDegree.set(nextId, nextDeg);
      if (nextDeg === 0 && (!allowedIds || allowedIds.has(nextId))) {
        queue.push(nextId);
      }
    }
  }
  return sorted;
}

interface CacheEntry {
  signature: string;
  result: NodeExecutionResult;
}

const executionCache = new Map<string, CacheEntry>();

export function clearExecutionCache() {
  executionCache.clear();
}

export async function executePipeline(options: {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  seed: number;
  stopAtNodeId?: string;
}): Promise<{
  results: Record<string, NodeExecutionResult>;
  logs: ExecutionLogEntry[];
  totalMs: number;
}> {
  const { nodes, edges, seed, stopAtNodeId } = options;
  const order = getTopologicalOrder(nodes, edges, stopAtNodeId);
  const results: Record<string, NodeExecutionResult> = {};
  const logs: ExecutionLogEntry[] = [];
  let totalMs = 0;

  // Ensure default Lena portrait sample is ready before processing
  const defaultLena = await getSampleImageDataAsync('portrait');

  const incomingEdgesByNode = new Map<string, PipelineEdge[]>();
  edges.forEach((e) => {
    const arr = incomingEdgesByNode.get(e.target) || [];
    arr.push(e);
    incomingEdgesByNode.set(e.target, arr);
  });

  for (const node of order) {
    const meta = NODE_CATALOG_MAP[node.type];
    const opName = node.type;
    logs.push({
      id: `${node.id}-start-${Date.now()}-${Math.random()}`,
      message: `Executing ${opName}`,
      kind: 'info',
      timestamp: Date.now(),
    });

    const incoming = incomingEdgesByNode.get(node.id) || [];
    const inputMap: Record<string, ImageData | undefined> = {};
    const inputSigParts: string[] = [];

    for (const edge of incoming) {
      const sourceRes = results[edge.source];
      if (sourceRes?.imageData) {
        inputMap[edge.targetHandle] = sourceRes.imageData;
        inputSigParts.push(
          `${edge.targetHandle}:${edge.source}:${sourceRes.durationMs}:${sourceRes.previewDataUrl.length}`
        );
      }
    }

    const signature = JSON.stringify({
      type: node.type,
      params: node.params,
      seed: meta?.isRandom ? seed : 0,
      inputs: inputSigParts.sort(),
    });

    const cached = executionCache.get(node.id);
    if (cached && cached.signature === signature) {
      const res: NodeExecutionResult = {
        ...cached.result,
        cached: true,
      };
      results[node.id] = res;
      totalMs += res.durationMs;
      logs.push({
        id: `${node.id}-done-${Date.now()}-${Math.random()}`,
        message: `Finished ${opName} in ${res.durationMs.toFixed(1)}ms (cache)`,
        kind: 'success',
        timestamp: Date.now(),
      });
      continue;
    }

    const t0 = performance.now();
    let outImage: ImageData;

    if (node.type === 'load_image') {
      const customUrl = String(node.params.customDataUrl || '');
      if (customUrl) {
        try {
          outImage = await loadImageDataFromDataUrl(customUrl);
        } catch {
          outImage = await getSampleImageDataAsync(String(node.params.sampleId || 'portrait'));
        }
      } else {
        outImage = await getSampleImageDataAsync(String(node.params.sampleId || 'portrait'));
      }
    } else {
      const primaryInput =
        inputMap.image || inputMap.a || inputMap.mask || Object.values(inputMap).find(Boolean) || defaultLena;

      switch (node.type) {
        case 'clahe':
        case 'albu_clahe': {
          const clip = Number(node.params.clipLimit ?? 2.5);
          const grid = Number(node.params.tileGridSize ?? 8);
          outImage = applyCLAHE(primaryInput, clip, grid);
          break;
        }
        case 'equalize_hist':
          outImage = applyEqualizeHist(primaryInput, node.params.luminanceOnly !== false);
          break;
        case 'brightness_contrast': {
          const bShift = Number(node.params.brightness ?? 15);
          const cMult = Number(node.params.contrast ?? 1.15);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < primaryInput.data.length; i += 4) {
            out[i] = Math.min(255, Math.max(0, Math.round((primaryInput.data[i] - 128) * cMult + 128 + bShift)));
            out[i + 1] = Math.min(255, Math.max(0, Math.round((primaryInput.data[i + 1] - 128) * cMult + 128 + bShift)));
            out[i + 2] = Math.min(255, Math.max(0, Math.round((primaryInput.data[i + 2] - 128) * cMult + 128 + bShift)));
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'gaussian_noise': {
          const sigma = Number(node.params.sigma ?? 22);
          const rand = seededRandom(seed + 17);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < primaryInput.data.length; i += 4) {
            const u1 = Math.max(1e-6, rand());
            const u2 = rand();
            const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
            out[i] = Math.min(255, Math.max(0, Math.round(primaryInput.data[i] + n)));
            out[i + 1] = Math.min(255, Math.max(0, Math.round(primaryInput.data[i + 1] + n)));
            out[i + 2] = Math.min(255, Math.max(0, Math.round(primaryInput.data[i + 2] + n)));
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'random_crop_resize': {
          const scale = Math.min(1, Math.max(0.35, Number(node.params.scale ?? 0.72)));
          const rand = seededRandom(seed + 31);
          const cw = Math.round(primaryInput.width * scale);
          const ch = Math.round(primaryInput.height * scale);
          const sx = Math.round(rand() * (primaryInput.width - cw));
          const sy = Math.round(rand() * (primaryInput.height - ch));
          const tmp = document.createElement('canvas');
          tmp.width = primaryInput.width;
          tmp.height = primaryInput.height;
          tmp.getContext('2d')!.putImageData(primaryInput, 0, 0);
          const dst = document.createElement('canvas');
          dst.width = primaryInput.width;
          dst.height = primaryInput.height;
          dst.getContext('2d')!.drawImage(tmp, sx, sy, cw, ch, 0, 0, primaryInput.width, primaryInput.height);
          outImage = dst.getContext('2d')!.getImageData(0, 0, primaryInput.width, primaryInput.height);
          break;
        }
        case 'channel_shuffle': {
          const orderStr = String(node.params.order || 'bgr');
          const out = new Uint8ClampedArray(primaryInput.data.length);
          const mapIdx =
            orderStr === 'grb'
              ? [1, 0, 2]
              : orderStr === 'brg'
                ? [2, 0, 1]
                : orderStr === 'rbg'
                  ? [0, 2, 1]
                  : [2, 1, 0];
          for (let i = 0; i < primaryInput.data.length; i += 4) {
            out[i] = primaryInput.data[i + mapIdx[0]];
            out[i + 1] = primaryInput.data[i + mapIdx[1]];
            out[i + 2] = primaryInput.data[i + mapIdx[2]];
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'flip_rotate': {
          const mode = String(node.params.mode || 'horizontal');
          const c = document.createElement('canvas');
          c.width = primaryInput.width;
          c.height = primaryInput.height;
          const ctx = c.getContext('2d')!;
          const tmp = document.createElement('canvas');
          tmp.width = primaryInput.width;
          tmp.height = primaryInput.height;
          tmp.getContext('2d')!.putImageData(primaryInput, 0, 0);
          ctx.save();
          if (mode === 'horizontal') {
            ctx.translate(c.width, 0);
            ctx.scale(-1, 1);
          } else if (mode === 'vertical') {
            ctx.translate(0, c.height);
            ctx.scale(1, -1);
          } else if (mode === 'rot90') {
            ctx.translate(c.width / 2, c.height / 2);
            ctx.rotate(Math.PI / 2);
            ctx.translate(-c.width / 2, -c.height / 2);
          } else {
            ctx.translate(c.width / 2, c.height / 2);
            ctx.rotate(Math.PI);
            ctx.translate(-c.width / 2, -c.height / 2);
          }
          ctx.drawImage(tmp, 0, 0);
          ctx.restore();
          outImage = ctx.getImageData(0, 0, c.width, c.height);
          break;
        }
        case 'to_hsv':
          outImage = applyToHSV(primaryInput, String(node.params.previewMode || 'hsv_channels'));
          break;
        case 'to_lab':
          outImage = applyToLAB(primaryInput, String(node.params.channelView || 'composite'));
          break;
        case 'colormap':
          outImage = applyColorMap(primaryInput, String(node.params.palette || 'turbo'));
          break;
        case 'split_channel': {
          const ch = String(node.params.channel || 'r');
          const off = ch === 'g' ? 1 : ch === 'b' ? 2 : 0;
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < out.length; i += 4) {
            const v = primaryInput.data[i + off];
            out[i] = out[i + 1] = out[i + 2] = v;
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'to_grayscale': {
          const gray = toGrayArray(primaryInput);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0, p = 0; i < out.length; i += 4, p++) {
            out[i] = out[i + 1] = out[i + 2] = gray[p];
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'in_range':
          outImage = applyInRange(primaryInput, node.params);
          break;
        case 'gaussian_blur':
          outImage = applyGaussianBlur(primaryInput, Number(node.params.kernelSize ?? 5));
          break;
        case 'median_blur':
          outImage = applyMedianBlur(primaryInput, Number(node.params.kernelSize ?? 5));
          break;
        case 'bilateral_filter':
          outImage = applyBilateralFilter(
            primaryInput,
            Number(node.params.diameter ?? 7),
            Number(node.params.sigmaColor ?? 55),
            Number(node.params.sigmaSpace ?? 35)
          );
          break;
        case 'box_blur':
          outImage = applyBoxBlur(primaryInput, Number(node.params.kernelSize ?? 5));
          break;
        case 'sobel':
          outImage = applySobel(primaryInput, String(node.params.axis || 'xy'), Number(node.params.gain ?? 1.0));
          break;
        case 'laplacian':
          outImage = applyLaplacian(primaryInput, Number(node.params.gain ?? 1.5));
          break;
        case 'sharpen': {
          const s = Number(node.params.strength ?? 1.2);
          const blurred = applyGaussianBlur(primaryInput, 5);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < out.length; i += 4) {
            for (let c = 0; c < 3; c++) {
              out[i + c] = Math.min(255, Math.max(0, Math.round(primaryInput.data[i + c] * (1 + s) - blurred.data[i + c] * s)));
            }
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'draw_histogram':
          outImage = applyDrawHistogram(
            primaryInput,
            String(node.params.mode || 'channels'),
            Number(node.params.plotWidth ?? 512),
            Number(node.params.plotHeight ?? 256)
          );
          break;
        case 'adaptive_threshold':
          outImage = applyAdaptiveThreshold(
            primaryInput,
            Number(node.params.blockSize ?? 11),
            Number(node.params.cConstant ?? 4)
          );
          break;
        case 'otsu_threshold':
          outImage = applyOtsuThreshold(primaryInput, Boolean(node.params.invert));
          break;
        case 'binary_threshold': {
          const th = Number(node.params.thresh ?? 128);
          const mode = String(node.params.type || 'binary');
          const gray = toGrayArray(primaryInput);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let p = 0, i = 0; p < gray.length; p++, i += 4) {
            const g = gray[p];
            let v = 0;
            if (mode === 'binary_inv') v = g > th ? 0 : 255;
            else if (mode === 'trunc') v = g > th ? th : g;
            else if (mode === 'tozero') v = g > th ? g : 0;
            else v = g > th ? 255 : 0;
            out[i] = out[i + 1] = out[i + 2] = v;
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'find_contours':
          outImage = applyFindContours(
            inputMap.mask || primaryInput,
            inputMap.canvas,
            Number(node.params.minArea ?? 40),
            node.params.drawBoxes !== false,
            String(node.params.color || '#22C55E')
          );
          break;
        case 'corner_detect':
          outImage = applyCornerDetect(
            primaryInput,
            Number(node.params.maxCorners ?? 45),
            Number(node.params.threshold ?? 35),
            String(node.params.color || '#F58220')
          );
          break;
        case 'canny':
          outImage = applyCanny(
            primaryInput,
            Number(node.params.threshold1 ?? 60),
            Number(node.params.threshold2 ?? 150)
          );
          break;
        case 'dilate':
          outImage = applyMorphology(
            primaryInput,
            'dilate',
            Number(node.params.kernelSize ?? 3),
            Number(node.params.iterations ?? 1)
          );
          break;
        case 'erode':
          outImage = applyMorphology(
            primaryInput,
            'erode',
            Number(node.params.kernelSize ?? 3),
            Number(node.params.iterations ?? 1)
          );
          break;
        case 'morph_open': {
          const k = Number(node.params.kernelSize ?? 3);
          outImage = applyMorphology(applyMorphology(primaryInput, 'erode', k, 1), 'dilate', k, 1);
          break;
        }
        case 'morph_close': {
          const k = Number(node.params.kernelSize ?? 5);
          outImage = applyMorphology(applyMorphology(primaryInput, 'dilate', k, 1), 'erode', k, 1);
          break;
        }
        case 'morph_gradient': {
          const k = Number(node.params.kernelSize ?? 3);
          const d = applyMorphology(primaryInput, 'dilate', k, 1);
          const e = applyMorphology(primaryInput, 'erode', k, 1);
          outImage = applyArithmetic(d, e, undefined, 'subtract', 1);
          break;
        }
        case 'arithmetic':
          outImage = applyArithmetic(
            inputMap.a || primaryInput,
            inputMap.b,
            inputMap.mask,
            String(node.params.operation || 'add'),
            Number(node.params.alpha ?? 0.65)
          );
          break;
        case 'apply_mask':
          outImage = applyMaskOp(
            inputMap.image || primaryInput,
            inputMap.mask,
            Boolean(node.params.invertMask)
          );
          break;
        case 'bitwise_not': {
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < out.length; i += 4) {
            out[i] = 255 - primaryInput.data[i];
            out[i + 1] = 255 - primaryInput.data[i + 1];
            out[i + 2] = 255 - primaryInput.data[i + 2];
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'gamma_correction': {
          const gamma = Math.max(0.1, Number(node.params.gamma ?? 1.35));
          const invGamma = 1 / gamma;
          const lut = new Uint8ClampedArray(256);
          for (let i = 0; i < 256; i++) lut[i] = Math.round(Math.pow(i / 255, invGamma) * 255);
          const out = new Uint8ClampedArray(primaryInput.data.length);
          for (let i = 0; i < out.length; i += 4) {
            out[i] = lut[primaryInput.data[i]];
            out[i + 1] = lut[primaryInput.data[i + 1]];
            out[i + 2] = lut[primaryInput.data[i + 2]];
            out[i + 3] = 255;
          }
          outImage = new ImageData(out, primaryInput.width, primaryInput.height);
          break;
        }
        case 'resize': {
          const s = Math.max(0.25, Math.min(2.0, Number(node.params.scale ?? 0.75)));
          outImage = resizeImageData(
            primaryInput,
            Math.max(32, Math.round(primaryInput.width * s)),
            Math.max(32, Math.round(primaryInput.height * s))
          );
          break;
        }
        case 'crop_roi': {
          const xPct = Number(node.params.x ?? 15) / 100;
          const yPct = Number(node.params.y ?? 15) / 100;
          const wPct = Number(node.params.w ?? 70) / 100;
          const hPct = Number(node.params.h ?? 70) / 100;
          const sx = Math.round(primaryInput.width * xPct);
          const sy = Math.round(primaryInput.height * yPct);
          const cw = Math.max(24, Math.min(primaryInput.width - sx, Math.round(primaryInput.width * wPct)));
          const ch = Math.max(24, Math.min(primaryInput.height - sy, Math.round(primaryInput.height * hPct)));
          const tmp = document.createElement('canvas');
          tmp.width = primaryInput.width;
          tmp.height = primaryInput.height;
          tmp.getContext('2d')!.putImageData(primaryInput, 0, 0);
          const dst = document.createElement('canvas');
          dst.width = cw;
          dst.height = ch;
          dst.getContext('2d')!.drawImage(tmp, sx, sy, cw, ch, 0, 0, cw, ch);
          outImage = dst.getContext('2d')!.getImageData(0, 0, cw, ch);
          break;
        }
        case 'warp_affine':
          outImage = applyWarpAffine(
            primaryInput,
            Number(node.params.angle ?? 18),
            Number(node.params.scale ?? 0.92),
            Number(node.params.tx ?? 0),
            Number(node.params.ty ?? 0),
            Number(node.params.shearX ?? 0),
            String(node.params.borderMode || 'constant')
          );
          break;
        case 'warp_perspective':
          outImage = applyWarpPerspective(primaryInput, node.params);
          break;
        case 'osd_canvas': {
          const elements = Array.isArray(node.params.elements) ? (node.params.elements as OSDElement[]) : [];
          outImage = renderOSDElementsToImageData(
            primaryInput,
            elements,
            Number(node.params.globalOpacity ?? 1),
            Boolean(node.params.showGridBurnIn)
          );
          break;
        }
        case 'osd_telemetry': {
          const hudColor = String(node.params.accentColor || '#22C55E');
          const title = String(node.params.label || 'VISION STUDIO // WASM');
          const fScale = Number(node.params.fontScale ?? 0.5);
          const els: OSDElement[] = [
            {
              id: 'hud-top',
              kind: 'text',
              name: 'HUD Label',
              visible: true,
              x: 0.04,
              y: 0.05,
              width: 0.6,
              height: 0.08,
              color: hudColor,
              thickness: 1.5,
              opacity: 0.95,
              text: `${title} (${primaryInput.width}x${primaryInput.height})`,
              fontScale: fScale,
              lineType: 'LINE_AA',
            },
          ];
          if (node.params.showCrosshair !== false) {
            els.push({
              id: 'hud-reticle',
              kind: 'reticle',
              name: 'HUD Crosshair',
              visible: true,
              x: 0.5,
              y: 0.5,
              width: 0.16,
              height: 0.16,
              color: hudColor,
              thickness: 1.5,
              opacity: 0.9,
              lineType: 'LINE_AA',
            });
          }
          outImage = renderOSDElementsToImageData(primaryInput, els, 1, true);
          break;
        }
        case 'osd_grid_ruler': {
          const div = Math.max(2, Math.min(8, Number(node.params.divisions ?? 3)));
          const color = String(node.params.gridColor || '#38BDF8');
          const els: OSDElement[] = [];
          for (let i = 1; i < div; i++) {
            const p = i / div;
            els.push({
              id: `grid-v-${i}`,
              kind: 'line',
              name: `V-${i}`,
              visible: true,
              x: p,
              y: 0,
              width: 0,
              height: 1,
              x2: p,
              y2: 1,
              color,
              thickness: 1,
              opacity: 0.65,
            });
            els.push({
              id: `grid-h-${i}`,
              kind: 'line',
              name: `H-${i}`,
              visible: true,
              x: 0,
              y: p,
              width: 1,
              height: 0,
              x2: 1,
              y2: p,
              color,
              thickness: 1,
              opacity: 0.65,
            });
          }
          if (node.params.showRuler !== false) {
            els.push({
              id: 'ruler-label',
              kind: 'text',
              name: 'Scale',
              visible: true,
              x: 0.04,
              y: 0.9,
              width: 0.4,
              height: 0.07,
              color,
              thickness: 1.2,
              opacity: 0.95,
              text: `${primaryInput.width}x${primaryInput.height}px`,
              fontScale: 0.46,
            });
          }
          outImage = renderOSDElementsToImageData(primaryInput, els, 1, true);
          break;
        }
        default:
          outImage = cloneImageData(primaryInput);
      }
    }

    const durationMs = Math.max(1.2, performance.now() - t0);
    totalMs += durationMs;

    const result: NodeExecutionResult = {
      nodeId: node.id,
      previewDataUrl: imageDataToDataUrl(outImage),
      width: outImage.width,
      height: outImage.height,
      channels:
        node.type === 'to_grayscale' ||
        node.type === 'canny' ||
        node.type === 'adaptive_threshold' ||
        node.type === 'otsu_threshold'
          ? 1
          : 3,
      durationMs,
      cached: false,
      imageData: outImage,
    };

    executionCache.set(node.id, { signature, result });
    results[node.id] = result;
    logs.push({
      id: `${node.id}-done-${Date.now()}-${Math.random()}`,
      message: `Finished ${opName} in ${durationMs.toFixed(1)}ms`,
      kind: 'success',
      timestamp: Date.now(),
    });
  }

  logs.push({
    id: `complete-${Date.now()}`,
    message: 'Execution complete',
    kind: 'info',
    timestamp: Date.now(),
  });

  return { results, logs, totalMs };
}
