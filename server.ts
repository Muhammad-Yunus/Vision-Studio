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
  target?: unknown;
  targetHandle?: unknown;
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
  const rawEdges = Array.isArray(raw.edges) ? (raw.edges as RawGeneratedEdge[]) : [];

  const nodes = [];
  const usedIds = new Set<string>();

  for (let i = 0; i < Math.min(24, rawNodes.length); i++) {
    const rn = rawNodes[i];
    const type = typeof rn.type === 'string' ? rn.type.trim() : '';
    const spec = NODE_CATALOG_MAP[type];
    if (!spec) continue;

    let id =
      typeof rn.id === 'string' && /^[a-zA-Z0-9_-]{1,40}$/.test(rn.id)
        ? rn.id
        : `n_${type}_${i + 1}`;
    if (usedIds.has(id)) id = `${id}_${i + 1}`;
    usedIds.add(id);

    const x =
      typeof rn.position?.x === 'number' && Number.isFinite(rn.position.x)
        ? Math.max(0, Math.min(2400, Math.round(rn.position.x)))
        : 40 + i * 210;
    const y =
      typeof rn.position?.y === 'number' && Number.isFinite(rn.position.y)
        ? Math.max(0, Math.min(1400, Math.round(rn.position.y)))
        : 180 + (i % 2) * 140;

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
  const edges = [];
  const usedTargets = new Set<string>();

  for (let i = 0; i < Math.min(48, rawEdges.length); i++) {
    const re = rawEdges[i];
    const source = typeof re.source === 'string' ? re.source : '';
    const target = typeof re.target === 'string' ? re.target : '';
    if (!nodeById.has(source) || !nodeById.has(target) || source === target) continue;

    const srcSpec = NODE_CATALOG_MAP[nodeById.get(source)!.type];
    const tgtSpec = NODE_CATALOG_MAP[nodeById.get(target)!.type];
    if (!srcSpec || !tgtSpec || srcSpec.outputs.length === 0 || tgtSpec.inputs.length === 0) continue;

    const sourceHandle =
      typeof re.sourceHandle === 'string' && srcSpec.outputs.some((o) => o.id === re.sourceHandle)
        ? re.sourceHandle
        : srcSpec.outputs[0].id;
    const targetHandle =
      typeof re.targetHandle === 'string' && tgtSpec.inputs.some((inp) => inp.id === re.targetHandle)
        ? re.targetHandle
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
      'Convert the user request into a valid DAG JSON object with keys: "name", "description", "nodes", "edges".',
      'Always start with a "load_image" node on the left (e.g., x: 40, y: 220) and lay out downstream nodes left-to-right (increment x by ~210px per stage, branch y by +/-170px).',
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
