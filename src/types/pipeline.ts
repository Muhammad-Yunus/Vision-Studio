export type NodeCategory =
  | 'augment'
  | 'color'
  | 'filters'
  | 'analysis'
  | 'math'
  | 'morphology'
  | 'geometric'
  | 'face'
  | 'ml'
  | 'io'
  | 'osd';

export interface PortDefinition {
  id: string;
  label: string;
  optional?: boolean;
}

export type ParamType =
  | 'number'
  | 'range'
  | 'select'
  | 'boolean'
  | 'text'
  | 'color'
  | 'image_source'
  | 'osd_layers'
  | 'webcam_control';

export interface ParamOption {
  label: string;
  value: string | number;
}

export interface ParamDefinition {
  key: string;
  label: string;
  type: ParamType;
  defaultValue: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: ParamOption[];
  description?: string;
}

export type OSDElementKind =
  | 'rect'
  | 'bbox_label'
  | 'circle'
  | 'line'
  | 'arrow'
  | 'text'
  | 'reticle';

export interface OSDElement {
  id: string;
  kind: OSDElementKind;
  name: string;
  visible: boolean;
  /** Coordinates in normalized 0..1 space or pixel space relative to canvas width/height */
  x: number;
  y: number;
  width: number;
  height: number;
  /** End point for line / arrow (relative to canvas width/height 0..1) */
  x2?: number;
  y2?: number;
  color: string;
  thickness: number;
  filled?: boolean;
  opacity: number;
  text?: string;
  fontScale?: number;
  lineType?: 'LINE_AA' | 'LINE_8' | 'LINE_4';
}

export interface NodeCatalogItem {
  type: string;
  title: string;
  category: NodeCategory;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
  isRandom?: boolean;
  transparentPreview?: boolean;
}

export interface PipelineNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  params: Record<string, unknown>;
  bypassed?: boolean;
}

export interface PipelineEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export interface NodeMetrics {
  count?: number;
  sourceNode?: string;
  unit?: string;
  totalArea?: number;
  ownedByNode?: boolean;
}

export interface NodeExecutionResult {
  nodeId: string;
  previewDataUrl: string;
  width: number;
  height: number;
  channels: number;
  durationMs: number;
  cached: boolean;
  bypassed?: boolean;
  imageData?: ImageData;
  inputPreviewDataUrl?: string;
  inputImageData?: ImageData;
  metrics?: NodeMetrics;
}

export interface BatchSampleResult {
  seed: number;
  previewDataUrl: string;
  width: number;
  height: number;
  durationMs: number;
}

export interface ExecutionLogEntry {
  id: string;
  message: string;
  kind: 'info' | 'success' | 'warn' | 'error';
  timestamp: number;
}

export interface PipelineTemplate {
  id: string;
  name: string;
  description: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  defaultSelectedNodeId?: string;
}
