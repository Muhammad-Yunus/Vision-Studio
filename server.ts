import 'dotenv/config';
import express from 'express';
import type { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { NODE_CATALOG_MAP } from './src/data/nodeCatalog.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const IS_PROD = process.env.NODE_ENV === 'production';

interface RawGeneratedNode {
  id?: unknown;
  type?: unknown;
  position?: { x?: unknown; y?: unknown };
  params?: Record<string, unknown>;
}

interface RawGeneratedEdge {
  id?: unknown;
  source?: unknown;
  sourceHandle?: unknown;
  sourcePort?: unknown;
  from?: unknown;
  fromPort?: unknown;
  src?: unknown;
  sourceNode?: unknown;
  from_node?: unknown;
  target?: unknown;
  targetHandle?: unknown;
  targetPort?: unknown;
  to?: unknown;
  toPort?: unknown;
  dst?: unknown;
  targetNode?: unknown;
  to_node?: unknown;
}

function getAIConfig() {
  const baseUrl = (process.env.AI_BASE_URL || 'https://api.openai.com/v1').trim();
  const model = (process.env.AI_MODEL || 'gpt-4o-mini').trim();
  const secret = (process.env.AI_SECRET || '').trim();
  return { baseUrl, model, secret };
}

function validateEndpointUrl(rawUrl: string): string {
  const parsed = new URL(rawUrl);
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error('AI_BASE_URL must use http or https protocol.');
  }
  const clean = parsed.toString().replace(/\/+$/, '');
  return clean.endsWith('/chat/completions') ? clean : `${clean}/chat/completions`;
}

function extractJsonObject(rawText: string): Record<string, unknown> {
  const trimmed = rawText.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  const firstBrace = candidate.indexOf('{');
  const lastBrace = candidate.lastIndexOf('}');
  if (firstBrace === -1 || lastBrace <= firstBrace) {
    throw new Error('Model response did not contain a valid JSON object.');
  }
  return JSON.parse(candidate.slice(firstBrace, lastBrace + 1)) as Record<string, unknown>;
}

function sanitizeGeneratedPipeline(raw: Record<string, unknown>) {
  const rawNodes = Array.isArray(raw.nodes) ? (raw.nodes as RawGeneratedNode[]) : [];
  const rawEdgesSource = Array.isArray(raw.edges)
    ? raw.edges
    : Array.isArray(raw.links)
      ? raw.links
      : Array.isArray(raw.connections)
        ? raw.connections
        : [];
  const rawEdges = rawEdgesSource as RawGeneratedEdge[];

  const nodes = [];
  const usedIds = new Set<string>();
  const rawRefToId = new Map<string, string>();
  const typeToIds = new Map<string, string[]>();

  for (let i = 0; i < Math.min(24, rawNodes.length); i++) {
    const rn = rawNodes[i];
    const type = typeof rn.type === 'string' ? rn.type.trim() : '';
    const spec = NODE_CATALOG_MAP[type];
    if (!spec) continue;

    const rawIdStr =
      typeof rn.id === 'string'
        ? rn.id.trim()
        : typeof rn.id === 'number' && Number.isFinite(rn.id)
          ? String(rn.id)
          : '';

    let id =
      rawIdStr && /^[a-zA-Z0-9_-]{1,40}$/.test(rawIdStr)
        ? rawIdStr
        : rawIdStr
          ? rawIdStr.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 40) || `n_${type}_${i + 1}`
          : `n_${type}_${i + 1}`;
    if (usedIds.has(id)) id = `${id}_${i + 1}`;
    usedIds.add(id);

    if (rawIdStr) rawRefToId.set(rawIdStr, id);
    rawRefToId.set(id, id);
    if (!rawRefToId.has(String(i))) rawRefToId.set(String(i), id);
    const tArr = typeToIds.get(type) || [];
    tArr.push(id);
    typeToIds.set(type, tArr);

    const x =
      typeof rn.position?.x === 'number' && Number.isFinite(rn.position.x)
        ? Math.max(0, Math.min(2400, Math.round(rn.position.x)))
        : 40 + i * 210;
    const y =
      typeof rn.position?.y === 'number' && Number.isFinite(rn.position.y)
        ? Math.max(0, Math.min(1400, Math.round(rn.position.y)))
        : 200;

    const params: Record<string, unknown> = {};
    for (const p of spec.params) {
      params[p.key] = structuredClone(p.defaultValue);
    }
    if (rn.params && typeof rn.params === 'object') {
      for (const p of spec.params) {
        if (Object.prototype.hasOwnProperty.call(rn.params, p.key)) {
          const val = rn.params[p.key];
          if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
            params[p.key] = val;
          } else if (p.key === 'elements' && Array.isArray(val)) {
            params[p.key] = val.slice(0, 15);
          }
        }
      }
    }

    nodes.push({ id, type, position: { x, y }, params });
  }

  if (nodes.length === 0) {
    throw new Error('AI generated zero valid OpenCV nodes.');
  }

  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const resolveNodeRef = (val: unknown): string => {
    const ref =
      typeof val === 'string'
        ? val.trim()
        : typeof val === 'number' && Number.isFinite(val)
          ? String(val)
          : '';
    if (!ref) return '';
    if (nodeById.has(ref)) return ref;
    const mapped = rawRefToId.get(ref);
    if (mapped && nodeById.has(mapped)) return mapped;
    const byType = typeToIds.get(ref);
    if (byType && byType.length === 1) return byType[0];
    return '';
  };

  const edges = [];
  const usedTargets = new Set<string>();

  for (let i = 0; i < Math.min(48, rawEdges.length); i++) {
    const re = rawEdges[i];
    const source = resolveNodeRef(re.source ?? re.from ?? re.src ?? re.sourceNode ?? re.from_node);
    const target = resolveNodeRef(re.target ?? re.to ?? re.dst ?? re.targetNode ?? re.to_node);
    if (!source || !target || source === target) continue;

    const srcSpec = NODE_CATALOG_MAP[nodeById.get(source)!.type];
    const tgtSpec = NODE_CATALOG_MAP[nodeById.get(target)!.type];
    if (!srcSpec || !tgtSpec || srcSpec.outputs.length === 0 || tgtSpec.inputs.length === 0) continue;

    const rawSrcHandle = re.sourceHandle ?? re.sourcePort ?? re.fromPort;
    const rawTgtHandle = re.targetHandle ?? re.targetPort ?? re.toPort;

    const sourceHandle =
      typeof rawSrcHandle === 'string' && srcSpec.outputs.some((o) => o.id === rawSrcHandle)
        ? rawSrcHandle
        : srcSpec.outputs[0].id;
    const targetHandle =
      typeof rawTgtHandle === 'string' && tgtSpec.inputs.some((inp) => inp.id === rawTgtHandle)
        ? rawTgtHandle
        : tgtSpec.inputs[0].id;

    const targetKey = `${target}:${targetHandle}`;
    if (usedTargets.has(targetKey)) continue;
    usedTargets.add(targetKey);

    edges.push({
      id: `e_ai_${i + 1}`,
      source,
      sourceHandle,
      target,
      targetHandle,
    });
  }

  // Fallback auto-wiring if model returned multiple nodes with no usable edge objects
  if (nodes.length > 1 && edges.length === 0) {
    const ordered = [...nodes].sort((a, b) => a.position.x - b.position.x);
    for (let i = 0; i < ordered.length - 1; i++) {
      const src = ordered[i];
      const tgt = ordered[i + 1];
      const srcSpec = NODE_CATALOG_MAP[src.type];
      const tgtSpec = NODE_CATALOG_MAP[tgt.type];
      if (!srcSpec || !tgtSpec || srcSpec.outputs.length === 0 || tgtSpec.inputs.length === 0) {
        continue;
      }
      edges.push({
        id: `e_ai_auto_${i + 1}`,
        source: src.id,
        sourceHandle: srcSpec.outputs[0].id,
        target: tgt.id,
        targetHandle: tgtSpec.inputs[0].id,
      });
    }
  }

  const name =
    typeof raw.name === 'string' && raw.name.trim()
      ? raw.name.trim().slice(0, 80)
      : 'AI Generated Pipeline';
  const description =
    typeof raw.description === 'string' && raw.description.trim()
      ? raw.description.trim().slice(0, 200)
      : 'Generated from prompt via OpenAI-compatible service.';

  return { name, description, nodes, edges };
}

function buildCatalogPromptSpec(): string {
  return Object.values(NODE_CATALOG_MAP)
    .map((n) => {
      const ins = n.inputs.map((i) => i.id).join(',') || 'none';
      const outs = n.outputs.map((o) => o.id).join(',') || 'none';
      const params = n.params.map((p) => `${p.key}=${JSON.stringify(p.defaultValue)}`).join(', ');
      return `- type: "${n.type}" (${n.category}) | inputs: [${ins}] | outputs: [${outs}] | params: {${params}}`;
    })
    .join('\n');
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '512kb' }));

  app.get('/api/ai/status', (_req: Request, res: Response) => {
    const { baseUrl, model, secret } = getAIConfig();
    res.json({
      configured: Boolean(secret && baseUrl),
      model,
      hasBaseUrl: Boolean(baseUrl),
    });
  });

  app.post('/api/ai/generate-pipeline', async (req: Request, res: Response) => {
    const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt.trim() : '';
    if (!prompt) {
      res.status(400).json({ error: 'Prompt is required.' });
      return;
    }
    if (prompt.length > 2000) {
      res.status(400).json({ error: 'Prompt must be 2000 characters or fewer.' });
      return;
    }

    const { baseUrl, model, secret } = getAIConfig();
    if (!secret) {
      res.status(503).json({
        error:
          'AI_SECRET is not configured on the server. Set AI_BASE_URL, AI_MODEL, and AI_SECRET in environment secrets.',
      });
      return;
    }

    let endpoint: string;
    try {
      endpoint = validateEndpointUrl(baseUrl);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Invalid AI_BASE_URL.' });
      return;
    }

    const systemPrompt = [
      'You are an OpenCV visual pipeline architect for Vision Studio.',
      'Convert the user request into a valid connected DAG JSON object with keys: "name", "description", "nodes", "edges".',
      'Always start with a "load_image" node on the left (x: 40, y: 200) and lay out downstream nodes left-to-right (increment x by ~210px per stage, branch y by +/-170px).',
      'CRITICAL: You MUST populate the "edges" array connecting every node using exact node "id" values and exact port IDs ("sourceHandle" from source outputs, "targetHandle" from target inputs).',
      'Example JSON structure:',
      '{"name":"Edge Pipeline","description":"...","nodes":[{"id":"n1","type":"load_image","position":{"x":40,"y":200},"params":{"sampleId":"portrait"}},{"id":"n2","type":"canny","position":{"x":250,"y":200},"params":{"threshold1":60,"threshold2":150}}],"edges":[{"id":"e1","source":"n1","sourceHandle":"image","target":"n2","targetHandle":"image"}]}',
      'Use ONLY node types and handle IDs from this exact catalog:',
      buildCatalogPromptSpec(),
      'Return ONLY raw JSON without markdown prose.',
    ].join('\n');

    try {
      const upstream = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${secret}`,
        },
        body: JSON.stringify({
          model,
          temperature: 0.2,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: prompt },
          ],
        }),
        signal: AbortSignal.timeout(30000),
      });

      const payloadText = await upstream.text();
      if (!upstream.ok) {
        res.status(upstream.status).json({
          error: `OpenAI-compatible upstream returned ${upstream.status}: ${payloadText.slice(0, 260)}`,
        });
        return;
      }

      const data = JSON.parse(payloadText) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const content = data?.choices?.[0]?.message?.content;
      if (!content) {
        res.status(502).json({ error: 'Upstream AI returned empty completion content.' });
        return;
      }

      const parsedJson = extractJsonObject(content);
      const pipeline = sanitizeGeneratedPipeline(parsedJson);
      res.json({ pipeline, model });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to generate pipeline.';
      res.status(500).json({ error: msg });
    }
  });

  if (!IS_PROD) {
    app.use(express.static(path.resolve(__dirname, 'public')));
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*path', (_req: Request, res: Response) => {
      const indexHtml = path.join(distDir, 'index.html');
      if (fs.existsSync(indexHtml)) {
        res.sendFile(indexHtml);
      } else {
        res.status(404).send('Build output not found. Run npm run build first.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Vision Studio server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
