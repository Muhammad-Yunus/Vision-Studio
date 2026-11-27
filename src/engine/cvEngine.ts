import { NODE_CATALOG_MAP } from '../data/nodeCatalog';
import {
  BatchSampleResult,
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

let resolvedCv: any = null;
let cvReadyPromise: Promise<any> | null = null;
let yunetModelPromise: Promise<boolean> | null = null;

export function isOpenCVWasmReady(): boolean {
  return Boolean(resolvedCv?.Mat || (typeof window !== 'undefined' && window.cv?.Mat));
}

export async function ensureOpenCVReady(): Promise<any> {
  if (resolvedCv && resolvedCv.Mat) return resolvedCv;
  if (typeof window !== 'undefined' && window.cv?.Mat) {
    resolvedCv = window.cv;
    return resolvedCv;
  }
  if (cvReadyPromise) return cvReadyPromise;

  cvReadyPromise = (async () => {
    if (typeof window === 'undefined') {
      throw new Error('OpenCV.js WASM requires a browser window');
    }

    const start = Date.now();
    while (!window.cv) {
      if (Date.now() - start > 20000) {
        throw new Error('Timed out waiting for /opencv/opencv.js script');
      }
      await new Promise((r) => setTimeout(r, 25));
    }

    const raw = window.cv;
    if (raw instanceof Promise || typeof raw.then === 'function') {
      resolvedCv = await raw;
      window.cv = resolvedCv;
      return resolvedCv;
    }

    if (raw.Mat) {
      resolvedCv = raw;
      return resolvedCv;
    }

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timed out waiting for OpenCV WASM runtime initialization'));
      }, 20000);
      const prev = raw.onRuntimeInitialized;
      raw.onRuntimeInitialized = () => {
        clearTimeout(timeout);
        prev?.();
        resolve();
      };
      const poll = setInterval(() => {
        if (raw.Mat) {
          clearInterval(poll);
          clearTimeout(timeout);
          resolve();
        }
      }, 25);
    });

    resolvedCv = raw;
    window.cv = resolvedCv;
    return resolvedCv;
  })();

  return cvReadyPromise;
}

async function ensureYuNetModel(cv: any): Promise<boolean> {
  if (yunetModelPromise) return yunetModelPromise;

  yunetModelPromise = (async () => {
    try {
      const resp = await fetch('/models/face_detection_yunet_2023mar.onnx');
      if (!resp.ok) return false;
      const buf = await resp.arrayBuffer();
      const bytes = new Uint8Array(buf);
      try {
        cv.FS_createDataFile('/', 'yunet.onnx', bytes, true, false, false);
      } catch {
        // File already exists in Emscripten virtual FS
      }
      return true;
    } catch {
      return false;
    }
  })();

  return yunetModelPromise;
}

function getCv(): any {
  const cv = resolvedCv || (typeof window !== 'undefined' ? window.cv : null);
  if (!cv || !cv.Mat) {
    throw new Error('OpenCV.js WASM runtime is not initialized');
  }
  return cv;
}

function seededRandom(seed: number): () => number {
  let s = (seed | 0) + 1234567;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) % 100000) / 100000;
  };
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  const num = parseInt(clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean, 16);
  if (Number.isNaN(num)) return [34, 197, 94];
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function imageDataToDataUrl(img: ImageData): string {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

function matToImageData(cv: any, mat: any): ImageData {
  const ch = mat.channels();
  if (ch === 4 && mat.type() === cv.CV_8UC4) {
    return new ImageData(new Uint8ClampedArray(mat.data), mat.cols, mat.rows);
  }
  const rgba = new cv.Mat();
  const u8 = new cv.Mat();
  try {
    let target = mat;
    if (mat.depth() !== cv.CV_8U) {
      mat.convertTo(u8, cv.CV_8U);
      target = u8;
    }
    if (ch === 1) {
      cv.cvtColor(target, rgba, cv.COLOR_GRAY2RGBA);
    } else if (ch === 3) {
      cv.cvtColor(target, rgba, cv.COLOR_RGB2RGBA);
    } else {
      target.copyTo(rgba);
    }
    return new ImageData(new Uint8ClampedArray(rgba.data), rgba.cols, rgba.rows);
  } finally {
    rgba.delete();
    u8.delete();
  }
}

function imageDataToRgbMat(cv: any, img: ImageData): any {
  const rgba = cv.matFromImageData(img);
  const rgb = new cv.Mat();
  try {
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    return rgb;
  } finally {
    rgba.delete();
  }
}

function imageDataToGrayMat(cv: any, img: ImageData): any {
  const rgba = cv.matFromImageData(img);
  const gray = new cv.Mat();
  try {
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    return gray;
  } finally {
    rgba.delete();
  }
}

function cloneImageDataViaCv(cv: any, img: ImageData): ImageData {
  const mat = cv.matFromImageData(img);
  try {
    return matToImageData(cv, mat);
  } finally {
    mat.delete();
  }
}

function resizeImageDataViaCv(cv: any, src: ImageData, targetW: number, targetH: number): ImageData {
  if (src.width === targetW && src.height === targetH) return src;
  const mat = cv.matFromImageData(src);
  const dst = new cv.Mat();
  try {
    cv.resize(mat, dst, new cv.Size(targetW, targetH), 0, 0, cv.INTER_LINEAR);
    return matToImageData(cv, dst);
  } finally {
    mat.delete();
    dst.delete();
  }
}

// ============================================================================
// COLOR & AUGMENTATION (100% window.cv)
// ============================================================================

function applyCLAHE(cv: any, src: ImageData, clipLimit = 2.5, tileSize = 8): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const lab = new cv.Mat();
  const planes = new cv.MatVector();
  const lOut = new cv.Mat();
  const clahe = new cv.CLAHE(
    Math.max(0.5, clipLimit),
    new cv.Size(Math.max(2, Math.round(tileSize)), Math.max(2, Math.round(tileSize)))
  );
  const outRgb = new cv.Mat();
  try {
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    cv.split(lab, planes);
    const lPlane = planes.get(0);
    clahe.apply(lPlane, lOut);
    lPlane.delete();
    planes.set(0, lOut);
    cv.merge(planes, lab);
    cv.cvtColor(lab, outRgb, cv.COLOR_Lab2RGB);
    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    lab.delete();
    planes.delete();
    lOut.delete();
    clahe.delete();
    outRgb.delete();
  }
}

function applyEqualizeHist(cv: any, src: ImageData, luminanceOnly = true): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const ycrcb = new cv.Mat();
  const planes = new cv.MatVector();
  const eq = new cv.Mat();
  const outRgb = new cv.Mat();
  try {
    if (luminanceOnly) {
      cv.cvtColor(rgb, ycrcb, cv.COLOR_RGB2YCrCb);
      cv.split(ycrcb, planes);
      const yPlane = planes.get(0);
      cv.equalizeHist(yPlane, eq);
      yPlane.delete();
      planes.set(0, eq);
      cv.merge(planes, ycrcb);
      cv.cvtColor(ycrcb, outRgb, cv.COLOR_YCrCb2RGB);
    } else {
      cv.split(rgb, planes);
      for (let i = 0; i < 3; i++) {
        const ch = planes.get(i);
        const chEq = new cv.Mat();
        cv.equalizeHist(ch, chEq);
        ch.delete();
        planes.set(i, chEq);
        chEq.delete();
      }
      cv.merge(planes, outRgb);
    }
    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    ycrcb.delete();
    planes.delete();
    eq.delete();
    outRgb.delete();
  }
}

function applyBrightnessContrast(cv: any, src: ImageData, contrast = 1.15, brightness = 15): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    // OpenCV convertTo: dst(x,y) = saturate_cast(alpha * src(x,y) + beta)
    const beta = 128 * (1 - contrast) + brightness;
    rgb.convertTo(dst, -1, contrast, beta);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applyGaussianNoise(cv: any, src: ImageData, sigma = 22, seed = 0): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const src16 = new cv.Mat();
  const noise = new cv.Mat(src.height, src.width, cv.CV_16SC3);
  const meanM = cv.matFromArray(1, 3, cv.CV_64F, [0, 0, 0]);
  const stdM = cv.matFromArray(1, 3, cv.CV_64F, [sigma, sigma, sigma]);
  const sum16 = new cv.Mat();
  const dst = new cv.Mat();
  try {
    cv.setRNGSeed(Math.abs((seed | 0) + 17));
    cv.randn(noise, meanM, stdM);
    rgb.convertTo(src16, cv.CV_16SC3);
    cv.add(src16, noise, sum16);
    sum16.convertTo(dst, cv.CV_8UC3);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    src16.delete();
    noise.delete();
    meanM.delete();
    stdM.delete();
    sum16.delete();
    dst.delete();
  }
}

function applyRandomCropResize(cv: any, src: ImageData, scale = 0.72, seed = 0): ImageData {
  const s = Math.min(1, Math.max(0.35, scale));
  const rand = seededRandom(seed + 31);
  const cw = Math.max(16, Math.round(src.width * s));
  const ch = Math.max(16, Math.round(src.height * s));
  const sx = Math.max(0, Math.min(src.width - cw, Math.round(rand() * (src.width - cw))));
  const sy = Math.max(0, Math.min(src.height - ch, Math.round(rand() * (src.height - ch))));

  const rgba = cv.matFromImageData(src);
  const roi = rgba.roi(new cv.Rect(sx, sy, cw, ch));
  const dst = new cv.Mat();
  try {
    cv.resize(roi, dst, new cv.Size(src.width, src.height), 0, 0, cv.INTER_LINEAR);
    return matToImageData(cv, dst);
  } finally {
    rgba.delete();
    roi.delete();
    dst.delete();
  }
}

function applyFlipRotate(cv: any, src: ImageData, mode = 'horizontal'): ImageData {
  const rgba = cv.matFromImageData(src);
  const dst = new cv.Mat();
  const resized = new cv.Mat();
  try {
    if (mode === 'horizontal') {
      cv.flip(rgba, dst, 1);
      return matToImageData(cv, dst);
    }
    if (mode === 'vertical') {
      cv.flip(rgba, dst, 0);
      return matToImageData(cv, dst);
    }
    if (mode === 'rot90') {
      cv.rotate(rgba, dst, cv.ROTATE_90_CLOCKWISE);
      return matToImageData(cv, dst);
    }
    cv.rotate(rgba, dst, cv.ROTATE_180);
    return matToImageData(cv, dst);
  } finally {
    rgba.delete();
    dst.delete();
    resized.delete();
  }
}

function applyChannelShuffle(cv: any, src: ImageData, orderStr = 'bgr'): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const planes = new cv.MatVector();
  const reordered = new cv.MatVector();
  const dst = new cv.Mat();
  const mapIdx =
    orderStr === 'grb'
      ? [1, 0, 2]
      : orderStr === 'brg'
        ? [2, 0, 1]
        : orderStr === 'rbg'
          ? [0, 2, 1]
          : [2, 1, 0];
  try {
    cv.split(rgb, planes);
    const chs = [planes.get(0), planes.get(1), planes.get(2)];
    reordered.push_back(chs[mapIdx[0]]);
    reordered.push_back(chs[mapIdx[1]]);
    reordered.push_back(chs[mapIdx[2]]);
    cv.merge(reordered, dst);
    chs.forEach((c) => c.delete());
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    planes.delete();
    reordered.delete();
    dst.delete();
  }
}

function applyToHSV(cv: any, src: ImageData, previewMode = 'hsv_channels'): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const hsv = new cv.Mat();
  const planes = new cv.MatVector();
  try {
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    if (previewMode === 'hue_only') {
      cv.split(hsv, planes);
      const h = planes.get(0);
      const hScaled = new cv.Mat();
      h.convertTo(hScaled, -1, 255 / 180, 0);
      const img = matToImageData(cv, hScaled);
      h.delete();
      hScaled.delete();
      return img;
    }
    if (previewMode === 'sat_only') {
      cv.split(hsv, planes);
      const s = planes.get(1);
      const img = matToImageData(cv, s);
      s.delete();
      return img;
    }
    return matToImageData(cv, hsv);
  } finally {
    rgb.delete();
    hsv.delete();
    planes.delete();
  }
}

function applyToLAB(cv: any, src: ImageData, channelView = 'composite'): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const lab = new cv.Mat();
  const planes = new cv.MatVector();
  try {
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    if (channelView === 'l_channel' || channelView === 'a_channel' || channelView === 'b_channel') {
      cv.split(lab, planes);
      const idx = channelView === 'l_channel' ? 0 : channelView === 'a_channel' ? 1 : 2;
      const ch = planes.get(idx);
      const img = matToImageData(cv, ch);
      ch.delete();
      return img;
    }
    return matToImageData(cv, lab);
  } finally {
    rgb.delete();
    lab.delete();
    planes.delete();
  }
}

function applyColorMap(cv: any, src: ImageData, palette = 'turbo'): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const cmBgr = new cv.Mat();
  const rgbaOut = new cv.Mat();
  const mapCode =
    palette === 'jet'
      ? cv.COLORMAP_JET
      : palette === 'hot'
        ? cv.COLORMAP_HOT
        : palette === 'inferno'
          ? cv.COLORMAP_INFERNO
          : palette === 'viridis'
            ? cv.COLORMAP_VIRIDIS
            : cv.COLORMAP_TURBO;
  try {
    cv.applyColorMap(gray, cmBgr, mapCode);
    cv.cvtColor(cmBgr, rgbaOut, cv.COLOR_BGR2RGBA);
    return matToImageData(cv, rgbaOut);
  } finally {
    gray.delete();
    cmBgr.delete();
    rgbaOut.delete();
  }
}

function applySplitChannel(cv: any, src: ImageData, channel = 'r'): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const planes = new cv.MatVector();
  try {
    cv.split(rgb, planes);
    const idx = channel === 'g' ? 1 : channel === 'b' ? 2 : 0;
    const plane = planes.get(idx);
    const img = matToImageData(cv, plane);
    plane.delete();
    return img;
  } finally {
    rgb.delete();
    planes.delete();
  }
}

function applyToGrayscale(cv: any, src: ImageData): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  try {
    return matToImageData(cv, gray);
  } finally {
    gray.delete();
  }
}

function applyInRange(cv: any, src: ImageData, params: Record<string, unknown>): ImageData {
  const lH = Number(params.lowerH ?? 0);
  const uH = Number(params.upperH ?? 180);
  const lS = Number(params.lowerS ?? 40);
  const uS = Number(params.upperS ?? 255);
  const lV = Number(params.lowerV ?? 40);
  const uV = Number(params.upperV ?? 255);

  const rgb = imageDataToRgbMat(cv, src);
  const low = new cv.Mat(rgb.rows, rgb.cols, rgb.type(), new cv.Scalar(lH, lS, lV));
  const high = new cv.Mat(rgb.rows, rgb.cols, rgb.type(), new cv.Scalar(uH, uS, uV));
  const mask = new cv.Mat();
  try {
    cv.inRange(rgb, low, high, mask);
    return matToImageData(cv, mask);
  } finally {
    rgb.delete();
    low.delete();
    high.delete();
    mask.delete();
  }
}

// ============================================================================
// FILTERS (100% window.cv)
// ============================================================================

function applyGaussianBlur(cv: any, src: ImageData, kernelSize = 5, sigmaX = 1.4): ImageData {
  const k = Math.max(1, Math.round(kernelSize) | 1);
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    cv.GaussianBlur(rgb, dst, new cv.Size(k, k), sigmaX, 0, cv.BORDER_DEFAULT);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applyMedianBlur(cv: any, src: ImageData, kernelSize = 5): ImageData {
  const k = Math.max(3, Math.round(kernelSize) | 1);
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    cv.medianBlur(rgb, dst, k);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applyBilateralFilter(
  cv: any,
  src: ImageData,
  diameter = 7,
  sigmaColor = 55,
  sigmaSpace = 35
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    cv.bilateralFilter(rgb, dst, Math.max(3, Math.round(diameter)), sigmaColor, sigmaSpace, cv.BORDER_DEFAULT);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applyBoxBlur(cv: any, src: ImageData, kernelSize = 5): ImageData {
  const k = Math.max(1, Math.round(kernelSize));
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    cv.blur(rgb, dst, new cv.Size(k, k), new cv.Point(-1, -1), cv.BORDER_DEFAULT);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applySobel(cv: any, src: ImageData, axis = 'xy', gain = 1.0): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const gx = new cv.Mat();
  const gy = new cv.Mat();
  const ax = new cv.Mat();
  const ay = new cv.Mat();
  const out = new cv.Mat();
  try {
    if (axis === 'x') {
      cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
      cv.convertScaleAbs(gx, out, gain, 0);
    } else if (axis === 'y') {
      cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
      cv.convertScaleAbs(gy, out, gain, 0);
    } else {
      cv.Sobel(gray, gx, cv.CV_16S, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
      cv.Sobel(gray, gy, cv.CV_16S, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
      cv.convertScaleAbs(gx, ax, gain, 0);
      cv.convertScaleAbs(gy, ay, gain, 0);
      cv.addWeighted(ax, 0.5, ay, 0.5, 0, out);
    }
    return matToImageData(cv, out);
  } finally {
    gray.delete();
    gx.delete();
    gy.delete();
    ax.delete();
    ay.delete();
    out.delete();
  }
}

function applyLaplacian(cv: any, src: ImageData, gain = 1.5): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const lap = new cv.Mat();
  const absLap = new cv.Mat();
  try {
    cv.Laplacian(gray, lap, cv.CV_16S, 3, 1, 0, cv.BORDER_DEFAULT);
    cv.convertScaleAbs(lap, absLap, gain, 0);
    return matToImageData(cv, absLap);
  } finally {
    gray.delete();
    lap.delete();
    absLap.delete();
  }
}

function applyCanny(cv: any, src: ImageData, t1 = 60, t2 = 150): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const edges = new cv.Mat();
  try {
    cv.Canny(gray, edges, Math.min(t1, t2), Math.max(t1, t2), 3, false);
    return matToImageData(cv, edges);
  } finally {
    gray.delete();
    edges.delete();
  }
}

function applySharpen(cv: any, src: ImageData, strength = 1.2): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const blurred = new cv.Mat();
  const dst = new cv.Mat();
  try {
    cv.GaussianBlur(rgb, blurred, new cv.Size(5, 5), 1.4, 0, cv.BORDER_DEFAULT);
    cv.addWeighted(rgb, 1 + strength, blurred, -strength, 0, dst);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    blurred.delete();
    dst.delete();
  }
}

// ============================================================================
// ANALYSIS (100% window.cv)
// ============================================================================

function applyDrawHistogram(cv: any, src: ImageData, mode = 'channels', plotW = 512, plotH = 256): ImageData {
  const w = Math.max(128, Math.min(512, Math.round(plotW)));
  const h = Math.max(128, Math.min(512, Math.round(plotH)));
  const rgb = imageDataToRgbMat(cv, src);
  const converted = new cv.Mat();
  const planes = new cv.MatVector();
  const canvas = new cv.Mat(h, w, cv.CV_8UC4, new cv.Scalar(0, 0, 0, 0));
  const emptyMask = new cv.Mat();

  const renderHistSeries = (
    planeMat: any,
    fillRgba: [number, number, number, number],
    strokeRgba: [number, number, number, number],
    maxBinScale: number
  ) => {
    const mv = new cv.MatVector();
    const hist = new cv.Mat();
    const coords = new Int32Array((256 + 2) * 2);
    mv.push_back(planeMat);
    cv.calcHist(mv, [0], emptyMask, hist, [256], [0, 256], false);

    const data = hist.data32F;
    coords[0] = 0;
    coords[1] = h - 1;
    for (let bin = 0; bin < 256; bin++) {
      const px = Math.round((bin / 255) * (w - 1));
      const norm = Math.min(1, data[bin] / Math.max(1, maxBinScale));
      const py = Math.max(2, Math.round(h - 1 - norm * (h * 0.88)));
      coords[(bin + 1) * 2] = px;
      coords[(bin + 1) * 2 + 1] = py;
    }
    coords[(256 + 1) * 2] = w - 1;
    coords[(256 + 1) * 2 + 1] = h - 1;

    const polyMat = cv.matFromArray(258, 1, cv.CV_32SC2, coords);
    const pv = new cv.MatVector();
    pv.push_back(polyMat);

    const layer = new cv.Mat(h, w, cv.CV_8UC4, new cv.Scalar(0, 0, 0, 0));
    cv.fillPoly(layer, pv, new cv.Scalar(...fillRgba));
    cv.polylines(layer, pv, false, new cv.Scalar(...strokeRgba), 2, cv.LINE_AA);
    cv.addWeighted(canvas, 1.0, layer, 0.72, 0, canvas);

    mv.delete();
    hist.delete();
    polyMat.delete();
    pv.delete();
    layer.delete();
  };

  try {
    if (mode === 'luminance') {
      cv.cvtColor(rgb, converted, cv.COLOR_RGB2GRAY);
      const mv = new cv.MatVector();
      const hist = new cv.Mat();
      mv.push_back(converted);
      cv.calcHist(mv, [0], emptyMask, hist, [256], [0, 256], false);
      const mm = cv.minMaxLoc(hist);
      mv.delete();
      hist.delete();
      renderHistSeries(converted, [226, 232, 240, 200], [248, 250, 252, 255], mm.maxVal);
    } else {
      if (mode === 'hsv') {
        cv.cvtColor(rgb, converted, cv.COLOR_RGB2HSV);
        cv.split(converted, planes);
      } else {
        cv.split(rgb, planes);
      }
      const chs = [planes.get(0), planes.get(1), planes.get(2)];
      let globalMax = 1;
      for (const ch of chs) {
        const mv = new cv.MatVector();
        const hist = new cv.Mat();
        mv.push_back(ch);
        cv.calcHist(mv, [0], emptyMask, hist, [256], [0, 256], false);
        for (let b = 2; b < 254; b++) {
          if (hist.data32F[b] > globalMax) globalMax = hist.data32F[b];
        }
        mv.delete();
        hist.delete();
      }
      renderHistSeries(chs[2], [59, 130, 246, 220], [96, 165, 250, 255], globalMax);
      renderHistSeries(chs[1], [34, 197, 94, 210], [74, 222, 128, 255], globalMax);
      renderHistSeries(chs[0], [239, 68, 68, 210], [248, 113, 113, 255], globalMax);
      chs.forEach((c) => c.delete());
    }
    return matToImageData(cv, canvas);
  } finally {
    rgb.delete();
    converted.delete();
    planes.delete();
    canvas.delete();
    emptyMask.delete();
  }
}

function applyAdaptiveThreshold(
  cv: any,
  src: ImageData,
  method = 'gaussian',
  blockSize = 11,
  cConst = 4
): ImageData {
  const bSize = Math.max(3, Math.round(blockSize) | 1);
  const gray = imageDataToGrayMat(cv, src);
  const dst = new cv.Mat();
  const adaptiveMethod =
    method === 'mean' ? cv.ADAPTIVE_THRESH_MEAN_C : cv.ADAPTIVE_THRESH_GAUSSIAN_C;
  try {
    cv.adaptiveThreshold(gray, dst, 255, adaptiveMethod, cv.THRESH_BINARY, bSize, cConst);
    return matToImageData(cv, dst);
  } finally {
    gray.delete();
    dst.delete();
  }
}

function applyOtsuThreshold(cv: any, src: ImageData, invert = false): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const dst = new cv.Mat();
  const flag = (invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY) + cv.THRESH_OTSU;
  try {
    cv.threshold(gray, dst, 0, 255, flag);
    return matToImageData(cv, dst);
  } finally {
    gray.delete();
    dst.delete();
  }
}

function applyBinaryThreshold(cv: any, src: ImageData, thresh = 128, mode = 'binary'): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const dst = new cv.Mat();
  const flag =
    mode === 'binary_inv'
      ? cv.THRESH_BINARY_INV
      : mode === 'trunc'
        ? cv.THRESH_TRUNC
        : mode === 'tozero'
          ? cv.THRESH_TOZERO
          : cv.THRESH_BINARY;
  try {
    cv.threshold(gray, dst, thresh, 255, flag);
    return matToImageData(cv, dst);
  } finally {
    gray.delete();
    dst.delete();
  }
}

function applyFindContours(
  cv: any,
  maskImg: ImageData,
  overlayImg: ImageData | undefined,
  minArea = 40,
  drawBoxes = true,
  color = '#22C55E'
): ImageData {
  const { width, height } = maskImg;
  const gray = imageDataToGrayMat(cv, maskImg);
  const bin = new cv.Mat();
  const contours = new cv.MatVector();
  const hier = new cv.Mat();
  let canvasMat: any;
  if (overlayImg) {
    const resized = resizeImageDataViaCv(cv, overlayImg, width, height);
    canvasMat = imageDataToRgbMat(cv, resized);
  } else {
    canvasMat = new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(9, 11, 14));
  }

  const [cr, cg, cb] = hexToRgb(color);
  try {
    cv.threshold(gray, bin, 127, 255, cv.THRESH_BINARY);
    cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let drawnBoxes = 0;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area >= minArea) {
        cv.drawContours(canvasMat, contours, i, new cv.Scalar(cr, cg, cb), 2, cv.LINE_AA);
        if (drawBoxes && drawnBoxes < 40) {
          const rect = cv.boundingRect(cnt);
          cv.rectangle(
            canvasMat,
            new cv.Point(rect.x, rect.y),
            new cv.Point(rect.x + rect.width, rect.y + rect.height),
            new cv.Scalar(245, 130, 32),
            1,
            cv.LINE_AA
          );
          drawnBoxes++;
        }
      }
      cnt.delete();
    }
    return matToImageData(cv, canvasMat);
  } finally {
    gray.delete();
    bin.delete();
    contours.delete();
    hier.delete();
    canvasMat.delete();
  }
}

function applyCornerDetect(
  cv: any,
  src: ImageData,
  maxCorners = 45,
  threshold = 35,
  color = '#F58220'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const gray = new cv.Mat();
  const gftt = new cv.GFTTDetector();
  const keypoints = new cv.KeyPointVector();
  const [cr, cg, cb] = hexToRgb(color);
  try {
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    gftt.setMaxFeatures(Math.max(1, Math.round(maxCorners)));
    gftt.setQualityLevel(Math.max(0.002, Math.min(0.25, threshold / 800)));
    gftt.setMinDistance(10);
    gftt.detect(gray, keypoints);

    for (let i = 0; i < keypoints.size(); i++) {
      const kp = keypoints.get(i);
      const pt = new cv.Point(Math.round(kp.pt.x), Math.round(kp.pt.y));
      cv.circle(rgb, pt, 4, new cv.Scalar(11, 13, 16), -1, cv.LINE_AA);
      cv.circle(rgb, pt, 3, new cv.Scalar(cr, cg, cb), -1, cv.LINE_AA);
    }
    return matToImageData(cv, rgb);
  } finally {
    rgb.delete();
    gray.delete();
    gftt.delete();
    keypoints.delete();
  }
}

// ============================================================================
// MATH & MORPHOLOGY (100% window.cv)
// ============================================================================

function applyArithmetic(
  cv: any,
  imgA: ImageData,
  imgB: ImageData | undefined,
  mask: ImageData | undefined,
  operation = 'add',
  alpha = 0.65
): ImageData {
  if (!imgB) return cloneImageDataViaCv(cv, imgA);
  const bResizedImg = resizeImageDataViaCv(cv, imgB, imgA.width, imgA.height);
  const aRgb = imageDataToRgbMat(cv, imgA);
  const bRgb = imageDataToRgbMat(cv, bResizedImg);
  const resRgb = new cv.Mat();
  const finalRgb = aRgb.clone();
  let maskBin: any = null;

  try {
    if (operation === 'subtract') cv.subtract(aRgb, bRgb, resRgb);
    else if (operation === 'absdiff') cv.absdiff(aRgb, bRgb, resRgb);
    else if (operation === 'bitwise_or') cv.bitwise_or(aRgb, bRgb, resRgb);
    else if (operation === 'bitwise_and') cv.bitwise_and(aRgb, bRgb, resRgb);
    else if (operation === 'bitwise_xor') cv.bitwise_xor(aRgb, bRgb, resRgb);
    else if (operation === 'blend') cv.addWeighted(aRgb, alpha, bRgb, 1 - alpha, 0, resRgb);
    else cv.add(aRgb, bRgb, resRgb);

    if (mask) {
      const mResized = resizeImageDataViaCv(cv, mask, imgA.width, imgA.height);
      const mGray = imageDataToGrayMat(cv, mResized);
      maskBin = new cv.Mat();
      cv.threshold(mGray, maskBin, 127, 255, cv.THRESH_BINARY);
      mGray.delete();
      resRgb.copyTo(finalRgb, maskBin);
      return matToImageData(cv, finalRgb);
    }

    return matToImageData(cv, resRgb);
  } finally {
    aRgb.delete();
    bRgb.delete();
    resRgb.delete();
    finalRgb.delete();
    if (maskBin) maskBin.delete();
  }
}

function applyMaskOp(cv: any, img: ImageData, mask: ImageData | undefined, invertMask = false): ImageData {
  if (!mask) return cloneImageDataViaCv(cv, img);
  const mResized = resizeImageDataViaCv(cv, mask, img.width, img.height);
  const rgb = imageDataToRgbMat(cv, img);
  const mGray = imageDataToGrayMat(cv, mResized);
  const mBin = new cv.Mat();
  const mInv = new cv.Mat();
  const out = new cv.Mat(img.height, img.width, cv.CV_8UC3, new cv.Scalar(0, 0, 0));
  try {
    cv.threshold(mGray, mBin, 127, 255, cv.THRESH_BINARY);
    if (invertMask) {
      cv.bitwise_not(mBin, mInv);
      rgb.copyTo(out, mInv);
    } else {
      rgb.copyTo(out, mBin);
    }
    return matToImageData(cv, out);
  } finally {
    rgb.delete();
    mGray.delete();
    mBin.delete();
    mInv.delete();
    out.delete();
  }
}

function applyBitwiseNot(cv: any, src: ImageData): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const dst = new cv.Mat();
  try {
    cv.bitwise_not(rgb, dst);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    dst.delete();
  }
}

function applyGammaCorrection(cv: any, src: ImageData, gamma = 1.35): ImageData {
  const g = Math.max(0.1, gamma);
  const invGamma = 1 / g;
  const rgb = imageDataToRgbMat(cv, src);
  const lut = new cv.Mat(1, 256, cv.CV_8UC1);
  const dst = new cv.Mat();
  try {
    for (let i = 0; i < 256; i++) {
      lut.data[i] = Math.min(255, Math.max(0, Math.round(Math.pow(i / 255, invGamma) * 255)));
    }
    cv.LUT(rgb, lut, dst);
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    lut.delete();
    dst.delete();
  }
}

function applyMorphology(
  cv: any,
  src: ImageData,
  mode: 'dilate' | 'erode' | 'open' | 'close' | 'gradient',
  kernelSize = 3,
  iterations = 1
): ImageData {
  const k = Math.max(1, Math.round(kernelSize) | 1);
  const iter = Math.max(1, Math.min(10, Math.round(iterations)));
  const rgb = imageDataToRgbMat(cv, src);
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k));
  const dst = new cv.Mat();
  try {
    if (mode === 'dilate') {
      cv.dilate(rgb, dst, kernel, new cv.Point(-1, -1), iter, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    } else if (mode === 'erode') {
      cv.erode(rgb, dst, kernel, new cv.Point(-1, -1), iter, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    } else if (mode === 'open') {
      cv.morphologyEx(rgb, dst, cv.MORPH_OPEN, kernel);
    } else if (mode === 'close') {
      cv.morphologyEx(rgb, dst, cv.MORPH_CLOSE, kernel);
    } else {
      cv.morphologyEx(rgb, dst, cv.MORPH_GRADIENT, kernel);
    }
    return matToImageData(cv, dst);
  } finally {
    rgb.delete();
    kernel.delete();
    dst.delete();
  }
}

// ============================================================================
// GEOMETRIC TRANSFORMATIONS (100% window.cv)
// ============================================================================

function applyWarpAffine(
  cv: any,
  src: ImageData,
  angleDeg = 18,
  scale = 0.92,
  tx = 0,
  ty = 0,
  shearX = 0,
  borderMode = 'constant'
): ImageData {
  const { width: w, height: h } = src;
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad) * scale;
  const sin = Math.sin(rad) * scale;
  const cx = w / 2;
  const cy = h / 2;

  const a = cos;
  const b = sin + shearX * cos;
  const d = -sin;
  const e = cos - shearX * sin;
  const c = cx - a * cx - b * cy + tx;
  const f = cy - d * cx - e * cy + ty;

  const srcMat = cv.matFromImageData(src);
  const dstMat = new cv.Mat();
  const M = cv.matFromArray(2, 3, cv.CV_64FC1, [a, b, c, d, e, f]);
  const bFlag =
    borderMode === 'reflect'
      ? cv.BORDER_REFLECT
      : borderMode === 'replicate'
        ? cv.BORDER_REPLICATE
        : cv.BORDER_CONSTANT;
  try {
    cv.warpAffine(srcMat, dstMat, M, new cv.Size(w, h), cv.INTER_LINEAR, bFlag, new cv.Scalar(0, 0, 0, 255));
    return matToImageData(cv, dstMat);
  } finally {
    srcMat.delete();
    dstMat.delete();
    M.delete();
  }
}

function applyWarpPerspective(cv: any, src: ImageData, params: Record<string, unknown>): ImageData {
  const { width: w, height: h } = src;
  const tlX = (Number(params.tlX ?? 12) / 100) * w;
  const tlY = (Number(params.tlY ?? 8) / 100) * h;
  const trX = (Number(params.trX ?? 88) / 100) * w;
  const trY = (Number(params.trY ?? 12) / 100) * h;
  const brX = (Number(params.brX ?? 98) / 100) * w;
  const brY = (Number(params.brY ?? 94) / 100) * h;
  const blX = (Number(params.blX ?? 2) / 100) * w;
  const blY = (Number(params.blY ?? 92) / 100) * h;
  const borderMode = String(params.borderMode || 'constant');

  const srcPts = [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1];
  const dstPts = [tlX, tlY, trX, trY, brX, brY, blX, blY];

  const srcMat = cv.matFromImageData(src);
  const dstMat = new cv.Mat();
  const sMat = cv.matFromArray(4, 1, cv.CV_32FC2, srcPts);
  const dMat = cv.matFromArray(4, 1, cv.CV_32FC2, dstPts);
  const M = cv.getPerspectiveTransform(sMat, dMat);
  const bFlag = borderMode === 'replicate' ? cv.BORDER_REPLICATE : cv.BORDER_CONSTANT;
  try {
    cv.warpPerspective(srcMat, dstMat, M, new cv.Size(w, h), cv.INTER_LINEAR, bFlag, new cv.Scalar(0, 0, 0, 255));
    return matToImageData(cv, dstMat);
  } finally {
    srcMat.delete();
    dstMat.delete();
    sMat.delete();
    dMat.delete();
    M.delete();
  }
}

// ============================================================================
// OPENCV FACE MODULE (cv.FaceDetectorYN + cv.inRange + cv.connectedComponentsWithStats)
// ============================================================================

interface DetectedFace {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  rightEye: [number, number];
  leftEye: [number, number];
  nose: [number, number];
  mouthRight: [number, number];
  mouthLeft: [number, number];
}

async function detectFacesWithOpenCV(
  cv: any,
  rgbMat: any,
  scoreThresh = 0.55
): Promise<DetectedFace[]> {
  const modelReady = await ensureYuNetModel(cv);
  const w = rgbMat.cols;
  const h = rgbMat.rows;

  if (modelReady && cv.FaceDetectorYN) {
    const bgr = new cv.Mat();
    const faces = new cv.Mat();
    let det: any = null;
    try {
      cv.cvtColor(rgbMat, bgr, cv.COLOR_RGB2BGR);
      det = new cv.FaceDetectorYN('/yunet.onnx', '', new cv.Size(w, h), scoreThresh, 0.3, 5000, 0, 0);
      det.detect(bgr, faces);

      const detected: DetectedFace[] = [];
      for (let i = 0; i < faces.rows; i++) {
        const off = i * 15;
        const d = faces.data32F;
        detected.push({
          x: Math.max(0, Math.round(d[off])),
          y: Math.max(0, Math.round(d[off + 1])),
          w: Math.max(8, Math.round(d[off + 2])),
          h: Math.max(8, Math.round(d[off + 3])),
          rightEye: [d[off + 4], d[off + 5]],
          leftEye: [d[off + 6], d[off + 7]],
          nose: [d[off + 8], d[off + 9]],
          mouthRight: [d[off + 10], d[off + 11]],
          mouthLeft: [d[off + 12], d[off + 13]],
          score: d[off + 14],
        });
      }
      if (detected.length > 0) return detected;
    } finally {
      bgr.delete();
      faces.delete();
      if (det) det.delete();
    }
  }

  // Fallback within OpenCV using YCrCb chrominance + connectedComponentsWithStats + minMaxLoc
  const ycrcb = new cv.Mat();
  const mask = new cv.Mat();
  const low = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(45, 133, 75));
  const high = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(255, 182, 132));
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
  const clean = new cv.Mat();
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const gray = new cv.Mat();

  try {
    cv.cvtColor(rgbMat, ycrcb, cv.COLOR_RGB2YCrCb);
    cv.inRange(ycrcb, low, high, mask);
    cv.morphologyEx(mask, clean, cv.MORPH_OPEN, kernel);
    const numLabels = cv.connectedComponentsWithStats(clean, labels, stats, centroids, 8, cv.CV_32S);

    let bestLabel = -1;
    let bestArea = w * h * 0.03;
    for (let lbl = 1; lbl < numLabels; lbl++) {
      const area = stats.intAt(lbl, cv.CC_STAT_AREA);
      if (area > bestArea) {
        bestArea = area;
        bestLabel = lbl;
      }
    }
    if (bestLabel === -1) return [];

    const rx = stats.intAt(bestLabel, cv.CC_STAT_LEFT);
    const ry = stats.intAt(bestLabel, cv.CC_STAT_TOP);
    const rw = Math.min(w - rx, stats.intAt(bestLabel, cv.CC_STAT_WIDTH));
    const rh = Math.min(h - ry, stats.intAt(bestLabel, cv.CC_STAT_HEIGHT));

    cv.cvtColor(rgbMat, gray, cv.COLOR_RGB2GRAY);
    const eyeY0 = Math.max(0, Math.round(ry + rh * 0.2));
    const eyeH = Math.max(8, Math.round(rh * 0.25));
    const halfW = Math.max(8, Math.round(rw * 0.35));

    const lRect = new cv.Rect(
      Math.max(0, Math.round(rx + rw * 0.14)),
      eyeY0,
      Math.min(w - Math.round(rx + rw * 0.14), halfW),
      Math.min(h - eyeY0, eyeH)
    );
    const rRect = new cv.Rect(
      Math.max(0, Math.round(rx + rw * 0.5)),
      eyeY0,
      Math.min(w - Math.round(rx + rw * 0.5), halfW),
      Math.min(h - eyeY0, eyeH)
    );

    const lRoi = gray.roi(lRect);
    const rRoi = gray.roi(rRect);
    const lLoc = cv.minMaxLoc(lRoi).minLoc;
    const rLoc = cv.minMaxLoc(rRoi).minLoc;
    lRoi.delete();
    rRoi.delete();

    const le: [number, number] = [lRect.x + lLoc.x, lRect.y + lLoc.y];
    const re: [number, number] = [rRect.x + rLoc.x, rRect.y + rLoc.y];
    const nose: [number, number] = [(le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5 + rh * 0.2];

    return [
      {
        x: rx,
        y: ry,
        w: rw,
        h: rh,
        score: Math.min(0.95, 0.72 + bestArea / (w * h)),
        rightEye: le,
        leftEye: re,
        nose,
        mouthRight: [nose[0] - rw * 0.14, nose[1] + rh * 0.18],
        mouthLeft: [nose[0] + rw * 0.14, nose[1] + rh * 0.18],
      },
    ];
  } finally {
    ycrcb.delete();
    mask.delete();
    low.delete();
    high.delete();
    kernel.delete();
    clean.delete();
    labels.delete();
    stats.delete();
    centroids.delete();
    gray.delete();
  }
}

async function applyFaceDetectHaar(
  cv: any,
  src: ImageData,
  scaleFactor = 1.1,
  minNeighbors = 4,
  privacyMode = 'bbox',
  boxColor = '#F43F5E'
): Promise<ImageData> {
  const rgb = imageDataToRgbMat(cv, src);
  const [cr, cg, cb] = hexToRgb(boxColor);
  try {
    const scoreThresh = Math.max(0.3, Math.min(0.85, 0.35 + minNeighbors * 0.045 - (scaleFactor - 1.1) * 0.2));
    const faces = await detectFacesWithOpenCV(cv, rgb, scoreThresh);

    for (const face of faces) {
      const fx = Math.max(0, Math.min(rgb.cols - 2, face.x));
      const fy = Math.max(0, Math.min(rgb.rows - 2, face.y));
      const fw = Math.max(4, Math.min(rgb.cols - fx, face.w));
      const fh = Math.max(4, Math.min(rgb.rows - fy, face.h));
      const rect = new cv.Rect(fx, fy, fw, fh);

      if (privacyMode === 'blur') {
        const roi = rgb.roi(rect);
        const blurred = new cv.Mat();
        cv.GaussianBlur(roi, blurred, new cv.Size(25, 25), 0, 0, cv.BORDER_DEFAULT);
        blurred.copyTo(roi);
        blurred.delete();
        roi.delete();
      } else if (privacyMode === 'pixelate') {
        const roi = rgb.roi(rect);
        const down = new cv.Mat();
        const pix = new cv.Mat();
        cv.resize(
          roi,
          down,
          new cv.Size(Math.max(4, Math.round(fw / 12)), Math.max(4, Math.round(fh / 12))),
          0,
          0,
          cv.INTER_AREA
        );
        cv.resize(down, pix, new cv.Size(fw, fh), 0, 0, cv.INTER_NEAREST);
        pix.copyTo(roi);
        down.delete();
        pix.delete();
        roi.delete();
      }

      cv.rectangle(rgb, new cv.Point(fx, fy), new cv.Point(fx + fw, fy + fh), new cv.Scalar(cr, cg, cb), 2, cv.LINE_AA);
      const label = `FACE ${face.score.toFixed(2)}`;
      const tagY = fy > 20 ? fy - 20 : fy;
      cv.rectangle(
        rgb,
        new cv.Point(fx, tagY),
        new cv.Point(Math.min(rgb.cols - 1, fx + label.length * 8 + 10), tagY + 18),
        new cv.Scalar(cr, cg, cb),
        -1
      );
      cv.putText(
        rgb,
        label,
        new cv.Point(fx + 4, tagY + 13),
        cv.FONT_HERSHEY_SIMPLEX,
        0.42,
        new cv.Scalar(11, 13, 16),
        1,
        cv.LINE_AA
      );
    }

    return matToImageData(cv, rgb);
  } finally {
    rgb.delete();
  }
}

async function applyFaceLandmarks(
  cv: any,
  src: ImageData,
  showMesh = true,
  pointRadius = 3,
  landmarkColor = '#38BDF8'
): Promise<ImageData> {
  const rgb = imageDataToRgbMat(cv, src);
  const [cr, cg, cb] = hexToRgb(landmarkColor);
  try {
    const faces = await detectFacesWithOpenCV(cv, rgb, 0.45);
    for (const face of faces) {
      const chin: [number, number] = [
        (face.mouthRight[0] + face.mouthLeft[0]) * 0.5,
        Math.min(rgb.rows - 4, face.y + face.h * 0.94),
      ];
      const pts: Array<[number, number]> = [
        face.rightEye,
        face.leftEye,
        face.nose,
        face.mouthRight,
        face.mouthLeft,
        chin,
      ];

      if (showMesh) {
        const meshCol = new cv.Scalar(240, 246, 252);
        const p = (pt: [number, number]) => new cv.Point(Math.round(pt[0]), Math.round(pt[1]));
        cv.line(rgb, p(face.rightEye), p(face.leftEye), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.rightEye), p(face.nose), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.leftEye), p(face.nose), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.nose), p(face.mouthRight), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.nose), p(face.mouthLeft), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.mouthRight), p(face.mouthLeft), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.mouthRight), p(chin), meshCol, 1, cv.LINE_AA);
        cv.line(rgb, p(face.mouthLeft), p(chin), meshCol, 1, cv.LINE_AA);
      }

      const rad = Math.max(2, Math.round(pointRadius));
      for (const [px, py] of pts) {
        const center = new cv.Point(Math.round(px), Math.round(py));
        cv.circle(rgb, center, rad + 1, new cv.Scalar(11, 13, 16), -1, cv.LINE_AA);
        cv.circle(rgb, center, rad, new cv.Scalar(cr, cg, cb), -1, cv.LINE_AA);
      }
    }
    return matToImageData(cv, rgb);
  } finally {
    rgb.delete();
  }
}

function applySkinMask(
  cv: any,
  src: ImageData,
  crMin = 133,
  crMax = 175,
  cbMin = 77,
  cbMax = 127,
  outputMode = 'mask'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const ycrcb = new cv.Mat();
  const low = new cv.Mat(rgb.rows, rgb.cols, rgb.type(), new cv.Scalar(40, crMin, cbMin));
  const high = new cv.Mat(rgb.rows, rgb.cols, rgb.type(), new cv.Scalar(255, crMax, cbMax));
  const mask = new cv.Mat();
  const dimmed = new cv.Mat();
  try {
    cv.cvtColor(rgb, ycrcb, cv.COLOR_RGB2YCrCb);
    cv.inRange(ycrcb, low, high, mask);
    if (outputMode === 'overlay') {
      rgb.convertTo(dimmed, -1, 0.2, 0);
      rgb.copyTo(dimmed, mask);
      return matToImageData(cv, dimmed);
    }
    return matToImageData(cv, mask);
  } finally {
    rgb.delete();
    ycrcb.delete();
    low.delete();
    high.delete();
    mask.delete();
    dimmed.delete();
  }
}

// ============================================================================
// OPENCV ML MODULE (cv.kmeans, cv.eigen + cv.gemm, cv.Mat CV_32F SVM kernel)
// ============================================================================

function applyKMeansQuantize(
  cv: any,
  src: ImageData,
  k = 5,
  attempts = 6,
  showPalette = true
): ImageData {
  const { width: w, height: h } = src;
  const clusters = Math.max(2, Math.min(16, Math.round(k)));
  const maxIter = Math.max(2, Math.min(15, Math.round(attempts)));
  const nPix = w * h;

  const rgb = imageDataToRgbMat(cv, src);
  const f32 = new cv.Mat();
  const samples = new cv.Mat(nPix, 3, cv.CV_32F);
  const labels = new cv.Mat();
  const centers = new cv.Mat();
  const outRgb = new cv.Mat(h, w, cv.CV_8UC3);

  try {
    rgb.convertTo(f32, cv.CV_32F);
    samples.data32F.set(f32.data32F);
    const crit = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, maxIter, 1.0);
    cv.kmeans(samples, clusters, labels, crit, 1, cv.KMEANS_PP_CENTERS, centers);

    const lblData = labels.data32S;
    const ctrData = centers.data32F;
    const outData = outRgb.data;
    for (let p = 0; p < nPix; p++) {
      const cIdx = lblData[p] * 3;
      const oIdx = p * 3;
      outData[oIdx] = Math.min(255, Math.max(0, Math.round(ctrData[cIdx])));
      outData[oIdx + 1] = Math.min(255, Math.max(0, Math.round(ctrData[cIdx + 1])));
      outData[oIdx + 2] = Math.min(255, Math.max(0, Math.round(ctrData[cIdx + 2])));
    }

    if (showPalette) {
      const stripH = 14;
      const bw = w / clusters;
      for (let c = 0; c < clusters; c++) {
        const cr = Math.min(255, Math.max(0, Math.round(ctrData[c * 3])));
        const cg = Math.min(255, Math.max(0, Math.round(ctrData[c * 3 + 1])));
        const cb = Math.min(255, Math.max(0, Math.round(ctrData[c * 3 + 2])));
        cv.rectangle(
          outRgb,
          new cv.Point(Math.round(c * bw), Math.max(0, h - stripH)),
          new cv.Point(Math.min(w - 1, Math.round((c + 1) * bw)), h - 1),
          new cv.Scalar(cr, cg, cb),
          -1
        );
      }
    }

    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    f32.delete();
    samples.delete();
    labels.delete();
    centers.delete();
    outRgb.delete();
  }
}

function applySVMClassifier(
  cv: any,
  src: ImageData,
  kernel = 'rbf',
  targetClass = 'warm_skin_tones',
  viewMode = 'overlay',
  cParam = 1.5
): ImageData {
  const { width: w, height: h } = src;
  const sharpness = Math.max(0.4, cParam * 1.1);

  const rgb = imageDataToRgbMat(cv, src);
  const fRgb = new cv.Mat();
  const hsv = new cv.Mat();
  const fHsv = new cv.Mat();
  const rgbP = new cv.MatVector();
  const hsvP = new cv.MatVector();
  const rawMargin = new cv.Mat(h, w, cv.CV_32FC1);
  const score = new cv.Mat(h, w, cv.CV_32FC1);
  const tmp1 = new cv.Mat();
  const tmp2 = new cv.Mat();
  const ones = new cv.Mat(h, w, cv.CV_32FC1, new cv.Scalar(1.0));
  const zeros = new cv.Mat(h, w, cv.CV_32FC1, new cv.Scalar(0.0));

  try {
    rgb.convertTo(fRgb, cv.CV_32FC3, 1 / 255);
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    hsv.convertTo(fHsv, cv.CV_32FC3, 1 / 255);

    cv.split(fRgb, rgbP);
    cv.split(fHsv, hsvP);
    const R = rgbP.get(0);
    const G = rgbP.get(1);
    const B = rgbP.get(2);
    const S = hsvP.get(1);
    const V = hsvP.get(2);

    if (targetClass === 'salient_lum') {
      const gray = new cv.Mat();
      const fGray = new cv.Mat();
      cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
      gray.convertTo(fGray, cv.CV_32FC1, 1 / 255);
      fGray.convertTo(rawMargin, cv.CV_32FC1, 2.4, -0.48 * 2.4);
      gray.delete();
      fGray.delete();
    } else if (targetClass === 'high_sat') {
      cv.addWeighted(S, 2.6, V, 0.4, -1.068, rawMargin);
    } else {
      // (R - B)*1.85 + (R - G)*0.95 + (V - 0.3)*0.35 - 0.28
      const rSubB = new cv.Mat();
      const rSubG = new cv.Mat();
      cv.subtract(R, B, rSubB);
      cv.subtract(R, G, rSubG);
      cv.addWeighted(rSubB, 1.85, rSubG, 0.95, 0, tmp1);
      cv.addWeighted(tmp1, 1.0, V, 0.35, -0.385, rawMargin);
      rSubB.delete();
      rSubG.delete();
    }

    if (kernel === 'linear') {
      rawMargin.convertTo(tmp1, cv.CV_32FC1, sharpness * 0.5, 0.5);
      cv.max(tmp1, zeros, tmp2);
      cv.min(tmp2, ones, score);
    } else if (kernel === 'poly') {
      // Polynomial-steepened sigmoid via cv.pow & cv.exp
      const sq = new cv.Mat();
      const mag = new cv.Mat();
      const posMask = new cv.Mat();
      const signF = new cv.Mat();
      cv.multiply(rawMargin, rawMargin, sq);
      cv.pow(sq, 0.7, mag);
      cv.threshold(rawMargin, posMask, 0, 2, cv.THRESH_BINARY);
      posMask.convertTo(signF, cv.CV_32FC1, 1, -1);
      cv.multiply(mag, signF, tmp1);
      tmp1.convertTo(tmp2, cv.CV_32FC1, -sharpness * 2.5, 0);
      cv.exp(tmp2, tmp1);
      cv.add(tmp1, ones, tmp2);
      cv.divide(ones, tmp2, score);
      sq.delete();
      mag.delete();
      posMask.delete();
      signF.delete();
    } else {
      rawMargin.convertTo(tmp1, cv.CV_32FC1, -sharpness * 2.8, 0);
      cv.exp(tmp1, tmp2);
      cv.add(tmp2, ones, tmp1);
      cv.divide(ones, tmp1, score);
    }

    R.delete();
    G.delete();
    B.delete();
    S.delete();
    V.delete();

    if (viewMode === 'mask') {
      const binF = new cv.Mat();
      const binU8 = new cv.Mat();
      cv.threshold(score, binF, 0.5, 255, cv.THRESH_BINARY);
      binF.convertTo(binU8, cv.CV_8UC1);
      const img = matToImageData(cv, binU8);
      binF.delete();
      binU8.delete();
      return img;
    }

    if (viewMode === 'heatmap') {
      const scoreU8 = new cv.Mat();
      const cmBgr = new cv.Mat();
      const rgbaOut = new cv.Mat();
      score.convertTo(scoreU8, cv.CV_8UC1, 255);
      cv.applyColorMap(scoreU8, cmBgr, cv.COLORMAP_TURBO);
      cv.cvtColor(cmBgr, rgbaOut, cv.COLOR_BGR2RGBA);
      const img = matToImageData(cv, rgbaOut);
      scoreU8.delete();
      cmBgr.delete();
      rgbaOut.delete();
      return img;
    }

    const maskF = new cv.Mat();
    const maskU8 = new cv.Mat();
    const dimmed = new cv.Mat();
    const highlighted = new cv.Mat();
    cv.threshold(score, maskF, 0.5, 255, cv.THRESH_BINARY);
    maskF.convertTo(maskU8, cv.CV_8UC1);
    rgb.convertTo(dimmed, -1, 0.28, 0);
    rgb.convertTo(highlighted, -1, 0.9, 28);
    highlighted.copyTo(dimmed, maskU8);
    const img = matToImageData(cv, dimmed);
    maskF.delete();
    maskU8.delete();
    dimmed.delete();
    highlighted.delete();
    return img;
  } finally {
    rgb.delete();
    fRgb.delete();
    hsv.delete();
    fHsv.delete();
    rgbP.delete();
    hsvP.delete();
    rawMargin.delete();
    score.delete();
    tmp1.delete();
    tmp2.delete();
    ones.delete();
    zeros.delete();
  }
}

function applyKNNMeanShift(
  cv: any,
  src: ImageData,
  spatialRadius = 4,
  colorRadius = 38
): ImageData {
  // Real spatial-color segmentation in OpenCV using cv.bilateralFilter + 5D spatial-color cv.kmeans
  const rgb = imageDataToRgbMat(cv, src);
  const smooth = new cv.Mat();
  const small = new cv.Mat();
  const labels = new cv.Mat();
  const centers = new cv.Mat();
  const clusteredSmall = new cv.Mat();
  const outRgb = new cv.Mat();

  try {
    const d = Math.max(3, Math.min(9, Math.round(spatialRadius) * 2 - 1));
    cv.bilateralFilter(rgb, smooth, d, colorRadius * 1.4, spatialRadius * 8, cv.BORDER_DEFAULT);

    // Downscale 0.5x for rapid 5D (x, y, R, G, B) cv.kmeans clustering, then upscale + refine
    const sw = Math.max(32, Math.round(src.width * 0.5));
    const sh = Math.max(32, Math.round(src.height * 0.5));
    cv.resize(smooth, small, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);

    const n = sw * sh;
    const samples5D = new cv.Mat(n, 5, cv.CV_32F);
    const spatialScale = Math.max(12, 90 - colorRadius) / Math.max(sw, sh);
    const sData = samples5D.data32F;
    const pData = small.data;

    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const p = y * sw + x;
        const p3 = p * 3;
        const p5 = p * 5;
        sData[p5] = pData[p3];
        sData[p5 + 1] = pData[p3 + 1];
        sData[p5 + 2] = pData[p3 + 2];
        sData[p5 + 3] = x * spatialScale;
        sData[p5 + 4] = y * spatialScale;
      }
    }

    const k = Math.max(6, Math.min(20, Math.round(28 - colorRadius * 0.22)));
    const crit = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 6, 1.0);
    cv.kmeans(samples5D, k, labels, crit, 1, cv.KMEANS_PP_CENTERS, centers);

    clusteredSmall.create(sh, sw, cv.CV_8UC3);
    const lblD = labels.data32S;
    const ctrD = centers.data32F;
    const csD = clusteredSmall.data;
    for (let p = 0; p < n; p++) {
      const cOff = lblD[p] * 5;
      const oOff = p * 3;
      csD[oOff] = Math.min(255, Math.max(0, Math.round(ctrD[cOff])));
      csD[oOff + 1] = Math.min(255, Math.max(0, Math.round(ctrD[cOff + 1])));
      csD[oOff + 2] = Math.min(255, Math.max(0, Math.round(ctrD[cOff + 2])));
    }

    cv.resize(clusteredSmall, outRgb, new cv.Size(src.width, src.height), 0, 0, cv.INTER_LINEAR);
    samples5D.delete();
    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    smooth.delete();
    small.delete();
    labels.delete();
    centers.delete();
    clusteredSmall.delete();
    outRgb.delete();
  }
}

function applyPCAEigen(cv: any, src: ImageData, components = 1, mode = 'reconstruct'): ImageData {
  const { width: w, height: h } = src;
  const nPix = w * h;
  const keep = Math.max(1, Math.min(3, Math.round(components)));

  const rgb = imageDataToRgbMat(cv, src);
  const fRgb = new cv.Mat();
  const meanTuple = cv.mean(rgb);
  const meanMat = new cv.Mat(h, w, cv.CV_32FC3, new cv.Scalar(meanTuple[0], meanTuple[1], meanTuple[2]));
  const centered = new cv.Mat();
  const X = new cv.Mat(nPix, 3, cv.CV_32F);
  const empty = new cv.Mat();
  const cov = new cv.Mat();
  const evals = new cv.Mat();
  const evecs = new cv.Mat();
  const proj = new cv.Mat();
  const recon = new cv.Mat();
  const reconGrid = new cv.Mat(h, w, cv.CV_32FC3);
  const shifted = new cv.Mat();
  const outU8 = new cv.Mat();

  try {
    rgb.convertTo(fRgb, cv.CV_32FC3);
    cv.subtract(fRgb, meanMat, centered);
    X.data32F.set(centered.data32F);

    cv.gemm(X, X, 1 / nPix, empty, 0, cov, cv.GEMM_1_T);
    cv.eigen(cov, evals, evecs);

    // Project samples onto eigenvectors: proj = X * evecs^T (N x 3)
    cv.gemm(X, evecs, 1, empty, 0, proj, cv.GEMM_2_T);

    if (mode === 'pc1' || mode === 'pc2') {
      const pcIdx = mode === 'pc2' ? 1 : 0;
      const gain = mode === 'pc2' ? 1.4 : 0.85;
      const grayF = new cv.Mat(h, w, cv.CV_32FC1);
      const pD = proj.data32F;
      const gD = grayF.data32F;
      for (let i = 0; i < nPix; i++) {
        gD[i] = Math.min(255, Math.max(0, 128 + pD[i * 3 + pcIdx] * gain));
      }
      grayF.convertTo(outU8, cv.CV_8UC1);
      grayF.delete();
      return matToImageData(cv, outU8);
    }

    // Zero out principal components >= keep before reconstruction
    if (keep < 3) {
      const pD = proj.data32F;
      for (let i = 0; i < nPix; i++) {
        const base = i * 3;
        if (keep < 2) pD[base + 1] = 0;
        if (keep < 3) pD[base + 2] = 0;
      }
    }

    cv.gemm(proj, evecs, 1, empty, 0, recon, 0);
    reconGrid.data32F.set(recon.data32F);
    cv.add(reconGrid, meanMat, shifted);
    shifted.convertTo(outU8, cv.CV_8UC3);
    return matToImageData(cv, outU8);
  } finally {
    rgb.delete();
    fRgb.delete();
    meanMat.delete();
    centered.delete();
    X.delete();
    empty.delete();
    cov.delete();
    evals.delete();
    evecs.delete();
    proj.delete();
    recon.delete();
    reconGrid.delete();
    shifted.delete();
    outU8.delete();
  }
}

// ============================================================================
// CLASSICAL FEATURE, FREQUENCY & GRABCUT NODES (100% window.cv)
// ============================================================================

function applyHoughLinesP(
  cv: any,
  src: ImageData,
  cannyLow = 50,
  cannyHigh = 150,
  threshold = 40,
  minLineLength = 28,
  maxLineGap = 10,
  overlayOnSource = true,
  lineColor = '#38BDF8'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  const lines = new cv.Mat();
  const outMat = overlayOnSource
    ? rgb.clone()
    : new cv.Mat(src.height, src.width, cv.CV_8UC3, new cv.Scalar(9, 11, 14));
  const [cr, cg, cb] = hexToRgb(lineColor);
  try {
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    cv.Canny(gray, edges, Math.min(cannyLow, cannyHigh), Math.max(cannyLow, cannyHigh), 3, false);
    cv.HoughLinesP(
      edges,
      lines,
      1,
      Math.PI / 180,
      Math.max(5, Math.round(threshold)),
      Math.max(2, minLineLength),
      Math.max(1, maxLineGap)
    );
    const data = lines.data32S;
    const count = Math.min(250, Math.floor(data.length / 4));
    for (let i = 0; i < count; i++) {
      const off = i * 4;
      cv.line(
        outMat,
        new cv.Point(data[off], data[off + 1]),
        new cv.Point(data[off + 2], data[off + 3]),
        new cv.Scalar(cr, cg, cb),
        2,
        cv.LINE_AA
      );
    }
    return matToImageData(cv, outMat);
  } finally {
    rgb.delete();
    gray.delete();
    edges.delete();
    lines.delete();
    outMat.delete();
  }
}

function applyHoughCircles(
  cv: any,
  src: ImageData,
  minDist = 24,
  param1 = 100,
  param2 = 28,
  minRadius = 6,
  maxRadius = 80,
  circleColor = '#F58220'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const circles = new cv.Mat();
  const [cr, cg, cb] = hexToRgb(circleColor);
  try {
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 1.4, 0, cv.BORDER_DEFAULT);
    cv.HoughCircles(
      blurred,
      circles,
      cv.HOUGH_GRADIENT,
      1,
      Math.max(4, minDist),
      Math.max(10, param1),
      Math.max(8, param2),
      Math.max(1, Math.round(minRadius)),
      Math.max(Math.round(minRadius) + 2, Math.round(maxRadius))
    );
    const data = circles.data32F;
    const count = Math.min(80, Math.floor(data.length / 3));
    for (let i = 0; i < count; i++) {
      const off = i * 3;
      const cx = Math.round(data[off]);
      const cy = Math.round(data[off + 1]);
      const r = Math.max(2, Math.round(data[off + 2]));
      cv.circle(rgb, new cv.Point(cx, cy), r, new cv.Scalar(cr, cg, cb), 2, cv.LINE_AA);
      cv.circle(rgb, new cv.Point(cx, cy), 2, new cv.Scalar(34, 197, 94), -1, cv.LINE_AA);
    }
    return matToImageData(cv, rgb);
  } finally {
    rgb.delete();
    gray.delete();
    blurred.delete();
    circles.delete();
  }
}

function applyOrbFeatures(
  cv: any,
  src: ImageData,
  maxFeatures = 250,
  drawRich = true,
  markerColor = '#10B981'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const gray = new cv.Mat();
  const orb = new cv.ORB(Math.max(10, Math.round(maxFeatures)));
  const kps = new cv.KeyPointVector();
  const emptyMask = new cv.Mat();
  const outRgb = rgb.clone();
  const [cr, cg, cb] = hexToRgb(markerColor);
  try {
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    orb.detect(gray, kps, emptyMask);
    const col = new cv.Scalar(cr, cg, cb, 255);
    if (cv.drawKeypoints && typeof cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS === 'number') {
      const flag = drawRich
        ? cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        : (cv.DrawMatchesFlags_DEFAULT ?? 0);
      cv.drawKeypoints(rgb, kps, outRgb, col, flag);
    } else {
      for (let i = 0; i < kps.size(); i++) {
        const kp = kps.get(i);
        const cx = Math.round(kp.pt.x);
        const cy = Math.round(kp.pt.y);
        const rad = drawRich ? Math.max(3, Math.min(18, Math.round((kp.size || 8) * 0.35))) : 3;
        cv.circle(outRgb, new cv.Point(cx, cy), rad, col, 1, cv.LINE_AA);
        if (drawRich && typeof kp.angle === 'number' && kp.angle >= 0) {
          const ang = (kp.angle * Math.PI) / 180;
          cv.line(
            outRgb,
            new cv.Point(cx, cy),
            new cv.Point(Math.round(cx + Math.cos(ang) * rad), Math.round(cy + Math.sin(ang) * rad)),
            col,
            1,
            cv.LINE_AA
          );
        }
      }
    }
    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    gray.delete();
    orb.delete();
    kps.delete();
    emptyMask.delete();
    outRgb.delete();
  }
}

function applyDftSpectrum(
  cv: any,
  src: ImageData,
  centerShift = true,
  colormap = 'inferno'
): ImageData {
  const gray = imageDataToGrayMat(cv, src);
  const optRows = cv.getOptimalDFTSize(gray.rows);
  const optCols = cv.getOptimalDFTSize(gray.cols);
  const padded = new cv.Mat();
  const floatGray = new cv.Mat();
  const planes = new cv.MatVector();
  const zeros = cv.Mat.zeros(optRows, optCols, cv.CV_32F);
  const complexI = new cv.Mat();
  const splitPlanes = new cv.MatVector();
  const mag = new cv.Mat();
  const ones = cv.Mat.ones(optRows, optCols, cv.CV_32F);
  const magCrop = new cv.Mat();
  const normU8 = new cv.Mat();
  const outResized = new cv.Mat();
  const cmBgr = new cv.Mat();
  const rgbaOut = new cv.Mat();

  try {
    cv.copyMakeBorder(
      gray,
      padded,
      0,
      optRows - gray.rows,
      0,
      optCols - gray.cols,
      cv.BORDER_CONSTANT,
      new cv.Scalar(0, 0, 0, 0)
    );
    padded.convertTo(floatGray, cv.CV_32F);
    planes.push_back(floatGray);
    planes.push_back(zeros);
    cv.merge(planes, complexI);
    cv.dft(complexI, complexI);
    cv.split(complexI, splitPlanes);

    const re = splitPlanes.get(0);
    const im = splitPlanes.get(1);
    cv.magnitude(re, im, mag);
    re.delete();
    im.delete();

    cv.add(mag, ones, mag);
    cv.log(mag, mag);

    const evenRows = mag.rows & -2;
    const evenCols = mag.cols & -2;
    const roiCrop = mag.roi(new cv.Rect(0, 0, evenCols, evenRows));
    roiCrop.copyTo(magCrop);
    roiCrop.delete();

    if (centerShift) {
      const cx = Math.floor(magCrop.cols / 2);
      const cy = Math.floor(magCrop.rows / 2);
      const q0 = magCrop.roi(new cv.Rect(0, 0, cx, cy));
      const q1 = magCrop.roi(new cv.Rect(cx, 0, cx, cy));
      const q2 = magCrop.roi(new cv.Rect(0, cy, cx, cy));
      const q3 = magCrop.roi(new cv.Rect(cx, cy, cx, cy));
      const tmp = new cv.Mat();
      q0.copyTo(tmp);
      q3.copyTo(q0);
      tmp.copyTo(q3);
      q1.copyTo(tmp);
      q2.copyTo(q1);
      tmp.copyTo(q2);
      tmp.delete();
      q0.delete();
      q1.delete();
      q2.delete();
      q3.delete();
    }

    cv.normalize(magCrop, normU8, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    cv.resize(normU8, outResized, new cv.Size(src.width, src.height), 0, 0, cv.INTER_LINEAR);

    if (colormap === 'gray') {
      return matToImageData(cv, outResized);
    }

    const code =
      colormap === 'turbo'
        ? cv.COLORMAP_TURBO
        : colormap === 'viridis'
          ? cv.COLORMAP_VIRIDIS
          : cv.COLORMAP_INFERNO;
    cv.applyColorMap(outResized, cmBgr, code);
    cv.cvtColor(cmBgr, rgbaOut, cv.COLOR_BGR2RGBA);
    return matToImageData(cv, rgbaOut);
  } finally {
    gray.delete();
    padded.delete();
    floatGray.delete();
    planes.delete();
    zeros.delete();
    complexI.delete();
    splitPlanes.delete();
    mag.delete();
    ones.delete();
    magCrop.delete();
    normU8.delete();
    outResized.delete();
    cmBgr.delete();
    rgbaOut.delete();
  }
}

function applyGrabCut(
  cv: any,
  src: ImageData,
  marginPct = 10,
  iterations = 2,
  outputMode = 'foreground'
): ImageData {
  const rgb = imageDataToRgbMat(cv, src);
  const maxDim = Math.max(src.width, src.height);
  const scale = maxDim > 220 ? 220 / maxDim : 1;
  const sw = Math.max(32, Math.round(src.width * scale));
  const sh = Math.max(32, Math.round(src.height * scale));

  const smallRgb = new cv.Mat();
  const gcMask = new cv.Mat(sh, sw, cv.CV_8UC1, new cv.Scalar(0));
  const bgdModel = new cv.Mat();
  const fgdModel = new cv.Mat();
  const fgBinSmall = new cv.Mat(sh, sw, cv.CV_8UC1, new cv.Scalar(0));
  const fgBinFull = new cv.Mat();
  const outRgb = new cv.Mat(src.height, src.width, cv.CV_8UC3, new cv.Scalar(9, 11, 14));

  try {
    if (scale < 1) {
      cv.resize(rgb, smallRgb, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
    } else {
      rgb.copyTo(smallRgb);
    }

    const m = Math.max(0.02, Math.min(0.4, marginPct / 100));
    const rx = Math.max(1, Math.round(sw * m));
    const ry = Math.max(1, Math.round(sh * m));
    const rw = Math.max(8, sw - rx * 2);
    const rh = Math.max(8, sh - ry * 2);
    const rect = new cv.Rect(rx, ry, rw, rh);

    cv.grabCut(
      smallRgb,
      gcMask,
      rect,
      bgdModel,
      fgdModel,
      Math.max(1, Math.min(5, Math.round(iterations))),
      cv.GC_INIT_WITH_RECT
    );

    const gData = gcMask.data;
    const fData = fgBinSmall.data;
    for (let i = 0; i < gData.length; i++) {
      // cv.GC_FGD = 1, cv.GC_PR_FGD = 3
      fData[i] = gData[i] === 1 || gData[i] === 3 ? 255 : 0;
    }

    if (scale < 1) {
      cv.resize(fgBinSmall, fgBinFull, new cv.Size(src.width, src.height), 0, 0, cv.INTER_NEAREST);
    } else {
      fgBinSmall.copyTo(fgBinFull);
    }

    if (outputMode === 'mask') {
      return matToImageData(cv, fgBinFull);
    }

    if (outputMode === 'overlay') {
      rgb.convertTo(outRgb, -1, 0.35, 0);
      rgb.copyTo(outRgb, fgBinFull);
      const bx = Math.round(src.width * m);
      const by = Math.round(src.height * m);
      const bw = Math.max(8, src.width - bx * 2);
      const bh = Math.max(8, src.height - by * 2);
      cv.rectangle(
        outRgb,
        new cv.Point(bx, by),
        new cv.Point(bx + bw, by + bh),
        new cv.Scalar(245, 130, 32),
        2,
        cv.LINE_AA
      );
      return matToImageData(cv, outRgb);
    }

    rgb.copyTo(outRgb, fgBinFull);
    return matToImageData(cv, outRgb);
  } finally {
    rgb.delete();
    smallRgb.delete();
    gcMask.delete();
    bgdModel.delete();
    fgdModel.delete();
    fgBinSmall.delete();
    fgBinFull.delete();
    outRgb.delete();
  }
}

function buildWebcamStandbyFrame(cv: any, w = 360, h = 270): ImageData {
  const mat = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(13, 17, 23));
  try {
    cv.rectangle(mat, new cv.Point(10, 10), new cv.Point(w - 10, h - 10), new cv.Scalar(42, 50, 62), 1, cv.LINE_AA);
    cv.circle(mat, new cv.Point(Math.round(w / 2), Math.round(h / 2) - 14), 26, new cv.Scalar(56, 158, 130), 2, cv.LINE_AA);
    cv.circle(mat, new cv.Point(Math.round(w / 2), Math.round(h / 2) - 14), 8, new cv.Scalar(245, 130, 32), -1, cv.LINE_AA);
    cv.putText(mat, 'WEBCAM STANDBY', new cv.Point(Math.round(w / 2) - 82, Math.round(h / 2) + 34), cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(230, 237, 243), 1, cv.LINE_AA);
    cv.putText(mat, 'Start camera in Inspector', new cv.Point(Math.round(w / 2) - 96, Math.round(h / 2) + 58), cv.FONT_HERSHEY_SIMPLEX, 0.42, new cv.Scalar(139, 148, 158), 1, cv.LINE_AA);
    return matToImageData(cv, mat);
  } finally {
    mat.delete();
  }
}

export function previewPerspectiveWithOpenCV(
  src: ImageData,
  params: Record<string, unknown>
): ImageData {
  const cv = getCv();
  return applyWarpPerspective(cv, src, params);
}

export function previewCropWithOpenCV(
  src: ImageData,
  params: Record<string, unknown>
): ImageData {
  const cv = getCv();
  const xPct = Number(params.x ?? 15) / 100;
  const yPct = Number(params.y ?? 15) / 100;
  const wPct = Number(params.w ?? 70) / 100;
  const hPct = Number(params.h ?? 70) / 100;
  const sx = Math.max(0, Math.min(src.width - 24, Math.round(src.width * xPct)));
  const sy = Math.max(0, Math.min(src.height - 24, Math.round(src.height * yPct)));
  const cw = Math.max(24, Math.min(src.width - sx, Math.round(src.width * wPct)));
  const ch = Math.max(24, Math.min(src.height - sy, Math.round(src.height * hPct)));
  const mat = cv.matFromImageData(src);
  const roi = mat.roi(new cv.Rect(sx, sy, cw, ch));
  const copied = roi.clone();
  try {
    return matToImageData(cv, copied);
  } finally {
    mat.delete();
    roi.delete();
    copied.delete();
  }
}

// ============================================================================
// OPENCV OSD DRAWING (100% window.cv)
// ============================================================================

export function renderOSDElementsToImageData(
  baseImg: ImageData,
  elements: OSDElement[],
  globalOpacity = 1.0,
  showGridBurnIn = false
): ImageData {
  const cv = getCv();
  const w = baseImg.width;
  const h = baseImg.height;
  const baseMat = cv.matFromImageData(baseImg);
  const overlayMat = baseMat.clone();

  try {
    if (showGridBurnIn) {
      const m = 12;
      const len = 18;
      const cCol = new cv.Scalar(240, 246, 252, 160);
      cv.line(overlayMat, new cv.Point(m, m + len), new cv.Point(m, m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(m, m), new cv.Point(m + len, m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(w - m - len, m), new cv.Point(w - m, m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(w - m, m), new cv.Point(w - m, m + len), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(m, h - m - len), new cv.Point(m, h - m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(m, h - m), new cv.Point(m + len, h - m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(w - m - len, h - m), new cv.Point(w - m, h - m), cCol, 1, cv.LINE_AA);
      cv.line(overlayMat, new cv.Point(w - m, h - m), new cv.Point(w - m, h - m - len), cCol, 1, cv.LINE_AA);
    }

    for (const el of elements) {
      if (!el.visible) continue;
      const [r, g, b] = hexToRgb(el.color || '#22C55E');
      const alphaScalar = Math.round(255 * Math.max(0.05, Math.min(1, (el.opacity ?? 1) * globalOpacity)));
      const col = new cv.Scalar(r, g, b, alphaScalar);
      const thickness = el.filled ? -1 : Math.max(1, Math.round(el.thickness || 2));
      const px = Math.round(el.x * w);
      const py = Math.round(el.y * h);
      const pw = Math.round(el.width * w);
      const ph = Math.round(el.height * h);

      if (el.kind === 'rect') {
        cv.rectangle(overlayMat, new cv.Point(px, py), new cv.Point(px + pw, py + ph), col, thickness, cv.LINE_AA);
      } else if (el.kind === 'bbox_label') {
        cv.rectangle(
          overlayMat,
          new cv.Point(px, py),
          new cv.Point(px + pw, py + ph),
          col,
          Math.max(1, Math.round(el.thickness || 2)),
          cv.LINE_AA
        );
        const label = el.text || 'OBJECT';
        const fScale = el.fontScale ?? 0.5;
        const tagH = Math.max(18, Math.round(22 * fScale * 1.4));
        cv.rectangle(
          overlayMat,
          new cv.Point(px, Math.max(0, py - tagH)),
          new cv.Point(Math.min(w - 1, px + Math.max(pw * 0.45, label.length * 9 + 14)), py),
          col,
          -1
        );
        cv.putText(
          overlayMat,
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
        cv.circle(
          overlayMat,
          new cv.Point(px + Math.round(pw / 2), py + Math.round(ph / 2)),
          rad,
          col,
          thickness,
          cv.LINE_AA
        );
      } else if (el.kind === 'line') {
        const x2 = Math.round((el.x2 ?? el.x + el.width) * w);
        const y2 = Math.round((el.y2 ?? el.y + el.height) * h);
        cv.line(
          overlayMat,
          new cv.Point(px, py),
          new cv.Point(x2, y2),
          col,
          Math.max(1, Math.round(el.thickness || 2)),
          cv.LINE_AA
        );
      } else if (el.kind === 'arrow') {
        const x2 = Math.round((el.x2 ?? el.x + el.width) * w);
        const y2 = Math.round((el.y2 ?? el.y + el.height) * h);
        cv.arrowedLine(
          overlayMat,
          new cv.Point(px, py),
          new cv.Point(x2, y2),
          col,
          Math.max(1, Math.round(el.thickness || 2)),
          cv.LINE_AA,
          0,
          0.22
        );
      } else if (el.kind === 'text') {
        const fScale = el.fontScale ?? 0.55;
        const txt = el.text || 'OSD TEXT';
        const tw = Math.round(txt.length * 14 * fScale);
        const th = Math.round(26 * fScale);
        cv.rectangle(
          overlayMat,
          new cv.Point(Math.max(0, px - 4), Math.max(0, py)),
          new cv.Point(Math.min(w - 1, px + tw + 8), Math.min(h - 1, py + th + 10)),
          new cv.Scalar(11, 13, 16, 200),
          -1
        );
        cv.putText(
          overlayMat,
          txt,
          new cv.Point(px, py + Math.round(22 * fScale)),
          cv.FONT_HERSHEY_SIMPLEX,
          fScale,
          col,
          Math.max(1, Math.round(el.thickness || 1.5)),
          cv.LINE_AA
        );
      } else if (el.kind === 'reticle') {
        const rad = Math.max(8, Math.round(Math.min(pw, ph) / 2));
        cv.circle(overlayMat, new cv.Point(px, py), rad, col, Math.max(1, Math.round(el.thickness || 1.5)), cv.LINE_AA);
        cv.line(overlayMat, new cv.Point(px - rad - 8, py), new cv.Point(px - 4, py), col, 1, cv.LINE_AA);
        cv.line(overlayMat, new cv.Point(px + 4, py), new cv.Point(px + rad + 8, py), col, 1, cv.LINE_AA);
        cv.line(overlayMat, new cv.Point(px, py - rad - 8), new cv.Point(px, py - 4), col, 1, cv.LINE_AA);
        cv.line(overlayMat, new cv.Point(px, py + 4), new cv.Point(px, py + rad + 8), col, 1, cv.LINE_AA);
      }
    }

    const alphaMix = Math.max(0.1, Math.min(1, globalOpacity));
    if (alphaMix < 0.99) {
      const blended = new cv.Mat();
      cv.addWeighted(overlayMat, alphaMix, baseMat, 1 - alphaMix, 0, blended);
      const img = matToImageData(cv, blended);
      blended.delete();
      return img;
    }

    return matToImageData(cv, overlayMat);
  } finally {
    baseMat.delete();
    overlayMat.delete();
  }
}

// ============================================================================
// PIPELINE SCHEDULER & EXECUTION
// ============================================================================

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
  webcamFrameData?: ImageData | null;
  skipCache?: boolean;
}): Promise<{
  results: Record<string, NodeExecutionResult>;
  logs: ExecutionLogEntry[];
  totalMs: number;
}> {
  const { nodes, edges, seed, stopAtNodeId, webcamFrameData, skipCache } = options;
  const order = getTopologicalOrder(nodes, edges, stopAtNodeId);
  const results: Record<string, NodeExecutionResult> = {};
  const logs: ExecutionLogEntry[] = [];
  let totalMs = 0;

  // Ensure OpenCV.js WASM and default Lena sample are loaded before running any node
  const [cv, defaultLena] = await Promise.all([
    ensureOpenCVReady(),
    getSampleImageDataAsync('portrait'),
  ]);

  const incomingEdgesByNode = new Map<string, PipelineEdge[]>();
  edges.forEach((e) => {
    const arr = incomingEdgesByNode.get(e.target) || [];
    arr.push(e);
    incomingEdgesByNode.set(e.target, arr);
  });

  for (const node of order) {
    const meta = NODE_CATALOG_MAP[node.type];
    const opName = node.type;
    const isBypassed = Boolean(node.bypassed);

    if (!isBypassed) {
      logs.push({
        id: `${node.id}-start-${Date.now()}-${Math.random()}`,
        message: `Executing ${opName} [cv.Mat]`,
        kind: 'info',
        timestamp: Date.now(),
      });
    }

    const incoming = incomingEdgesByNode.get(node.id) || [];
    const inputMap: Record<string, ImageData | undefined> = {};
    const inputSourceIdMap: Record<string, string | undefined> = {};
    const inputSigParts: string[] = [];

    for (const edge of incoming) {
      const sourceRes = results[edge.source];
      if (sourceRes?.imageData) {
        inputMap[edge.targetHandle] = sourceRes.imageData;
        inputSourceIdMap[edge.targetHandle] = edge.source;
        inputSigParts.push(
          `${edge.targetHandle}:${edge.source}:${sourceRes.durationMs}:${sourceRes.previewDataUrl.length}`
        );
      }
    }

    const primaryInputSourceId =
      inputSourceIdMap.image ||
      inputSourceIdMap.a ||
      inputSourceIdMap.mask ||
      Object.values(inputSourceIdMap).find(Boolean);
    const inputPreviewDataUrl = primaryInputSourceId
      ? results[primaryInputSourceId]?.previewDataUrl
      : undefined;
    const primaryInput =
      inputMap.image ||
      inputMap.a ||
      inputMap.mask ||
      Object.values(inputMap).find(Boolean) ||
      defaultLena;

    const isLiveWebcam = node.type === 'webcam_input' && Boolean(webcamFrameData);

    const signature = JSON.stringify({
      type: node.type,
      params: node.params,
      bypassed: isBypassed,
      seed: meta?.isRandom ? seed : 0,
      inputs: inputSigParts.sort(),
    });

    const cached = !skipCache && !isLiveWebcam ? executionCache.get(node.id) : undefined;
    if (cached && cached.signature === signature) {
      const res: NodeExecutionResult = {
        ...cached.result,
        cached: true,
        inputPreviewDataUrl: inputPreviewDataUrl ?? cached.result.inputPreviewDataUrl,
        inputImageData: primaryInputSourceId ? primaryInput : cached.result.inputImageData,
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

    if (isBypassed && node.type !== 'load_image' && node.type !== 'webcam_input') {
      outImage = cloneImageDataViaCv(cv, primaryInput);
    } else if (node.type === 'load_image') {
      const customUrl = String(node.params.customDataUrl || '');
      let rawImg: ImageData;
      if (customUrl) {
        try {
          rawImg = await loadImageDataFromDataUrl(customUrl);
        } catch {
          rawImg = await getSampleImageDataAsync(String(node.params.sampleId || 'portrait'));
        }
      } else {
        rawImg = await getSampleImageDataAsync(String(node.params.sampleId || 'portrait'));
      }
      // Marshal through cv.matFromImageData -> cv.Mat
      outImage = cloneImageDataViaCv(cv, rawImg);
    } else if (node.type === 'webcam_input') {
      if (webcamFrameData) {
        outImage = cloneImageDataViaCv(cv, webcamFrameData);
      } else {
        const resParts = String(node.params.resolution || '360x270').split('x').map(Number);
        outImage = buildWebcamStandbyFrame(cv, resParts[0] || 360, resParts[1] || 270);
      }
    } else {
      switch (node.type) {
        case 'clahe':
        case 'albu_clahe':
          outImage = applyCLAHE(
            cv,
            primaryInput,
            Number(node.params.clipLimit ?? 2.5),
            Number(node.params.tileGridSize ?? 8)
          );
          break;
        case 'equalize_hist':
          outImage = applyEqualizeHist(cv, primaryInput, node.params.luminanceOnly !== false);
          break;
        case 'brightness_contrast':
          outImage = applyBrightnessContrast(
            cv,
            primaryInput,
            Number(node.params.contrast ?? 1.15),
            Number(node.params.brightness ?? 15)
          );
          break;
        case 'gaussian_noise':
          outImage = applyGaussianNoise(cv, primaryInput, Number(node.params.sigma ?? 22), seed);
          break;
        case 'random_crop_resize':
          outImage = applyRandomCropResize(cv, primaryInput, Number(node.params.scale ?? 0.72), seed);
          break;
        case 'channel_shuffle':
          outImage = applyChannelShuffle(cv, primaryInput, String(node.params.order || 'bgr'));
          break;
        case 'flip_rotate':
          outImage = applyFlipRotate(cv, primaryInput, String(node.params.mode || 'horizontal'));
          break;
        case 'to_hsv':
          outImage = applyToHSV(cv, primaryInput, String(node.params.previewMode || 'hsv_channels'));
          break;
        case 'to_lab':
          outImage = applyToLAB(cv, primaryInput, String(node.params.channelView || 'composite'));
          break;
        case 'colormap':
          outImage = applyColorMap(cv, primaryInput, String(node.params.palette || 'turbo'));
          break;
        case 'split_channel':
          outImage = applySplitChannel(cv, primaryInput, String(node.params.channel || 'r'));
          break;
        case 'to_grayscale':
          outImage = applyToGrayscale(cv, primaryInput);
          break;
        case 'in_range':
          outImage = applyInRange(cv, primaryInput, node.params);
          break;
        case 'gaussian_blur':
          outImage = applyGaussianBlur(
            cv,
            primaryInput,
            Number(node.params.kernelSize ?? 5),
            Number(node.params.sigmaX ?? 1.4)
          );
          break;
        case 'median_blur':
          outImage = applyMedianBlur(cv, primaryInput, Number(node.params.kernelSize ?? 5));
          break;
        case 'bilateral_filter':
          outImage = applyBilateralFilter(
            cv,
            primaryInput,
            Number(node.params.diameter ?? 7),
            Number(node.params.sigmaColor ?? 55),
            Number(node.params.sigmaSpace ?? 35)
          );
          break;
        case 'box_blur':
          outImage = applyBoxBlur(cv, primaryInput, Number(node.params.kernelSize ?? 5));
          break;
        case 'sobel':
          outImage = applySobel(cv, primaryInput, String(node.params.axis || 'xy'), Number(node.params.gain ?? 1.0));
          break;
        case 'laplacian':
          outImage = applyLaplacian(cv, primaryInput, Number(node.params.gain ?? 1.5));
          break;
        case 'canny':
          outImage = applyCanny(
            cv,
            primaryInput,
            Number(node.params.threshold1 ?? 60),
            Number(node.params.threshold2 ?? 150)
          );
          break;
        case 'sharpen':
          outImage = applySharpen(cv, primaryInput, Number(node.params.strength ?? 1.2));
          break;
        case 'draw_histogram':
          outImage = applyDrawHistogram(
            cv,
            primaryInput,
            String(node.params.mode || 'channels'),
            Number(node.params.plotWidth ?? 512),
            Number(node.params.plotHeight ?? 256)
          );
          break;
        case 'adaptive_threshold':
          outImage = applyAdaptiveThreshold(
            cv,
            primaryInput,
            String(node.params.method || 'gaussian'),
            Number(node.params.blockSize ?? 11),
            Number(node.params.cConstant ?? 4)
          );
          break;
        case 'otsu_threshold':
          outImage = applyOtsuThreshold(cv, primaryInput, Boolean(node.params.invert));
          break;
        case 'binary_threshold':
          outImage = applyBinaryThreshold(
            cv,
            primaryInput,
            Number(node.params.thresh ?? 128),
            String(node.params.type || 'binary')
          );
          break;
        case 'find_contours':
          outImage = applyFindContours(
            cv,
            inputMap.mask || primaryInput,
            inputMap.canvas,
            Number(node.params.minArea ?? 40),
            node.params.drawBoxes !== false,
            String(node.params.color || '#22C55E')
          );
          break;
        case 'corner_detect':
          outImage = applyCornerDetect(
            cv,
            primaryInput,
            Number(node.params.maxCorners ?? 45),
            Number(node.params.threshold ?? 35),
            String(node.params.color || '#F58220')
          );
          break;
        case 'hough_lines':
          outImage = applyHoughLinesP(
            cv,
            primaryInput,
            Number(node.params.cannyLow ?? 50),
            Number(node.params.cannyHigh ?? 150),
            Number(node.params.threshold ?? 40),
            Number(node.params.minLineLength ?? 28),
            Number(node.params.maxLineGap ?? 10),
            node.params.overlayOnSource !== false,
            String(node.params.lineColor || '#38BDF8')
          );
          break;
        case 'hough_circles':
          outImage = applyHoughCircles(
            cv,
            primaryInput,
            Number(node.params.minDist ?? 24),
            Number(node.params.param1 ?? 100),
            Number(node.params.param2 ?? 28),
            Number(node.params.minRadius ?? 6),
            Number(node.params.maxRadius ?? 80),
            String(node.params.circleColor || '#F58220')
          );
          break;
        case 'orb_features':
          outImage = applyOrbFeatures(
            cv,
            primaryInput,
            Number(node.params.maxFeatures ?? 250),
            node.params.drawRich !== false,
            String(node.params.markerColor || '#10B981')
          );
          break;
        case 'dft_spectrum':
          outImage = applyDftSpectrum(
            cv,
            primaryInput,
            node.params.centerShift !== false,
            String(node.params.colormap || 'inferno')
          );
          break;
        case 'grabcut':
          outImage = applyGrabCut(
            cv,
            primaryInput,
            Number(node.params.marginPct ?? 10),
            Number(node.params.iterations ?? 2),
            String(node.params.outputMode || 'foreground')
          );
          break;
        case 'dilate':
          outImage = applyMorphology(
            cv,
            primaryInput,
            'dilate',
            Number(node.params.kernelSize ?? 3),
            Number(node.params.iterations ?? 1)
          );
          break;
        case 'erode':
          outImage = applyMorphology(
            cv,
            primaryInput,
            'erode',
            Number(node.params.kernelSize ?? 3),
            Number(node.params.iterations ?? 1)
          );
          break;
        case 'morph_open':
          outImage = applyMorphology(cv, primaryInput, 'open', Number(node.params.kernelSize ?? 3), 1);
          break;
        case 'morph_close':
          outImage = applyMorphology(cv, primaryInput, 'close', Number(node.params.kernelSize ?? 5), 1);
          break;
        case 'morph_gradient':
          outImage = applyMorphology(cv, primaryInput, 'gradient', Number(node.params.kernelSize ?? 3), 1);
          break;
        case 'arithmetic':
          outImage = applyArithmetic(
            cv,
            inputMap.a || primaryInput,
            inputMap.b,
            inputMap.mask,
            String(node.params.operation || 'add'),
            Number(node.params.alpha ?? 0.65)
          );
          break;
        case 'apply_mask':
          outImage = applyMaskOp(
            cv,
            inputMap.image || primaryInput,
            inputMap.mask,
            Boolean(node.params.invertMask)
          );
          break;
        case 'bitwise_not':
          outImage = applyBitwiseNot(cv, primaryInput);
          break;
        case 'gamma_correction':
          outImage = applyGammaCorrection(cv, primaryInput, Number(node.params.gamma ?? 1.35));
          break;
        case 'resize': {
          const s = Math.max(0.25, Math.min(2.0, Number(node.params.scale ?? 0.75)));
          outImage = resizeImageDataViaCv(
            cv,
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
          const sx = Math.max(0, Math.min(primaryInput.width - 24, Math.round(primaryInput.width * xPct)));
          const sy = Math.max(0, Math.min(primaryInput.height - 24, Math.round(primaryInput.height * yPct)));
          const cw = Math.max(24, Math.min(primaryInput.width - sx, Math.round(primaryInput.width * wPct)));
          const ch = Math.max(24, Math.min(primaryInput.height - sy, Math.round(primaryInput.height * hPct)));
          const mat = cv.matFromImageData(primaryInput);
          const roi = mat.roi(new cv.Rect(sx, sy, cw, ch));
          const copied = roi.clone();
          outImage = matToImageData(cv, copied);
          mat.delete();
          roi.delete();
          copied.delete();
          break;
        }
        case 'warp_affine':
          outImage = applyWarpAffine(
            cv,
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
          outImage = applyWarpPerspective(cv, primaryInput, node.params);
          break;
        case 'face_detect_haar':
          outImage = await applyFaceDetectHaar(
            cv,
            primaryInput,
            Number(node.params.scaleFactor ?? 1.1),
            Number(node.params.minNeighbors ?? 4),
            String(node.params.privacyMode || 'bbox'),
            String(node.params.boxColor || '#F43F5E')
          );
          break;
        case 'face_landmarks':
          outImage = await applyFaceLandmarks(
            cv,
            primaryInput,
            node.params.showMesh !== false,
            Number(node.params.pointRadius ?? 3),
            String(node.params.landmarkColor || '#38BDF8')
          );
          break;
        case 'face_skin_mask':
          outImage = applySkinMask(
            cv,
            primaryInput,
            Number(node.params.crMin ?? 133),
            Number(node.params.crMax ?? 175),
            Number(node.params.cbMin ?? 77),
            Number(node.params.cbMax ?? 127),
            String(node.params.outputMode || 'mask')
          );
          break;
        case 'ml_kmeans':
          outImage = applyKMeansQuantize(
            cv,
            primaryInput,
            Number(node.params.k ?? 5),
            Number(node.params.attempts ?? 6),
            node.params.showPalette !== false
          );
          break;
        case 'ml_svm_segment':
          outImage = applySVMClassifier(
            cv,
            primaryInput,
            String(node.params.kernel || 'rbf'),
            String(node.params.targetClass || 'warm_skin_tones'),
            String(node.params.viewMode || 'overlay'),
            Number(node.params.cParam ?? 1.5)
          );
          break;
        case 'ml_knn_posterize':
          outImage = applyKNNMeanShift(
            cv,
            primaryInput,
            Number(node.params.spatialRadius ?? 4),
            Number(node.params.colorRadius ?? 38)
          );
          break;
        case 'ml_pca':
          outImage = applyPCAEigen(
            cv,
            primaryInput,
            Number(node.params.components ?? 1),
            String(node.params.mode || 'reconstruct')
          );
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
          throw new Error(`Unsupported OpenCV pipeline node type: ${node.type}`);
      }
    }

    const durationMs = isBypassed ? 0.2 : Math.max(0.5, performance.now() - t0);
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
        node.type === 'otsu_threshold' ||
        node.type === 'binary_threshold'
          ? 1
          : 3,
      durationMs,
      cached: false,
      bypassed: isBypassed,
      imageData: outImage,
      inputPreviewDataUrl,
      inputImageData: primaryInputSourceId ? primaryInput : undefined,
    };

    if (!isLiveWebcam && !skipCache) {
      executionCache.set(node.id, { signature, result });
    }
    results[node.id] = result;
    logs.push({
      id: `${node.id}-done-${Date.now()}-${Math.random()}`,
      message: isBypassed
        ? `Bypassed ${opName} (pass-through)`
        : `Finished ${opName} in ${durationMs.toFixed(1)}ms`,
      kind: isBypassed ? 'warn' : 'success',
      timestamp: Date.now(),
    });
  }

  logs.push({
    id: `complete-${Date.now()}`,
    message: 'Execution complete (OpenCV.js WASM)',
    kind: 'info',
    timestamp: Date.now(),
  });

  return { results, logs, totalMs };
}

export async function executeBatchSamples(options: {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  baseSeed: number;
  count: number;
  targetNodeId?: string | null;
}): Promise<BatchSampleResult[]> {
  const { nodes, edges, baseSeed, count, targetNodeId } = options;
  const clamped = Math.max(1, Math.min(16, Math.round(count)));
  const order = getTopologicalOrder(nodes, edges);
  const resolvedTargetId = targetNodeId || order[order.length - 1]?.id;
  if (!resolvedTargetId) return [];

  const batch: BatchSampleResult[] = [];
  for (let i = 0; i < clamped; i++) {
    const s = baseSeed + i;
    const exec = await executePipeline({
      nodes,
      edges,
      seed: s,
      stopAtNodeId: targetNodeId || undefined,
    });
    const res = exec.results[resolvedTargetId];
    if (res) {
      batch.push({
        seed: s,
        previewDataUrl: res.previewDataUrl,
        width: res.width,
        height: res.height,
        durationMs: exec.totalMs,
      });
    }
  }
  return batch;
}
