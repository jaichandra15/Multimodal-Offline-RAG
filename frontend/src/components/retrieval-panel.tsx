'use client';

import { useState } from 'react';
import {
  Search, ArrowDown, Layers, Timer, Zap, ChevronDown, ChevronUp,
  FileText, ArrowRightLeft, X, Minimize2, Maximize2,
} from 'lucide-react';
import type { RetrievalTrace, RetrievalChunkTrace } from '@/lib/types';

// ── Score Bar ────────────────────────────────────────────────────────────────
function ScoreBar({ score, maxScore = 1, color }: { score: number; maxScore?: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (score / maxScore) * 100));
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] font-mono text-muted-foreground w-12 text-right">
        {score.toFixed(4)}
      </span>
    </div>
  );
}

// ── Chunk Card ───────────────────────────────────────────────────────────────
function ChunkCard({
  chunk,
  index,
  barColor,
  maxScore,
}: {
  chunk: RetrievalChunkTrace;
  index: number;
  barColor: string;
  maxScore: number;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="group rounded-lg border border-border/60 bg-background hover:border-border transition-colors">
      <button
        className="w-full flex items-start gap-2 p-2 text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="flex-shrink-0 h-5 w-5 rounded-md bg-violet-500/10 text-violet-600 dark:text-violet-400 flex items-center justify-center text-[10px] font-bold mt-0.5">
          {index}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-medium truncate leading-tight">
            {chunk.doc_title}
          </p>
          <ScoreBar score={chunk.score} maxScore={maxScore} color={barColor} />
        </div>
      </button>
      {expanded && (
        <div className="px-2 pb-2 pt-0">
          <p className="text-[10px] text-muted-foreground leading-relaxed bg-muted/40 rounded-md p-2 font-mono break-words">
            {chunk.preview}…
          </p>
        </div>
      )}
    </div>
  );
}

// ── Timeline Step ────────────────────────────────────────────────────────────
function PipelineStep({
  icon,
  title,
  timeMs,
  active,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  timeMs?: number | null;
  active?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="relative pl-6">
      {/* Timeline connector */}
      <div className="absolute left-[9px] top-6 bottom-0 w-px bg-border" />
      {/* Dot */}
      <div
        className={`absolute left-0 top-0.5 h-[18px] w-[18px] rounded-full flex items-center justify-center border-2 ${
          active
            ? 'bg-violet-500 border-violet-400 shadow-sm shadow-violet-500/40'
            : 'bg-muted border-border'
        }`}
      >
        <div className="text-white">{icon}</div>
      </div>
      <div className="pb-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-semibold">{title}</span>
          {timeMs != null && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
              {timeMs.toFixed(0)}ms
            </span>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────
export function RetrievalPipelinePanel({
  trace,
  visible,
  onClose,
}: {
  trace: RetrievalTrace | null;
  visible: boolean;
  onClose: () => void;
}) {
  const [minimized, setMinimized] = useState(false);

  if (!visible || !trace) return null;

  const maxPreScore = Math.max(...trace.pre_rerank.map((c) => Math.abs(c.score)), 0.001);
  const maxPostScore = Math.max(...trace.post_rerank.map((c) => Math.abs(c.score)), 0.001);

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 bg-background/95 backdrop-blur-xl border border-border rounded-2xl shadow-2xl shadow-black/20 transition-all duration-300 ${
        minimized ? 'w-64' : 'w-[380px] max-h-[75vh]'
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b bg-gradient-to-r from-violet-500/5 to-indigo-500/5 rounded-t-2xl">
        <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-1 rounded-lg">
          <Layers className="h-3 w-3 text-white" />
        </div>
        <h3 className="text-xs font-semibold flex-1">Retrieval Pipeline</h3>

        {/* Timing badge */}
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-violet-500/10 text-violet-600 dark:text-violet-400 flex items-center gap-1">
          <Timer className="h-2.5 w-2.5" />
          {trace.total_ms.toFixed(0)}ms
        </span>

        <button
          onClick={() => setMinimized((v) => !v)}
          className="h-5 w-5 rounded-md hover:bg-muted flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
        >
          {minimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
        </button>
        <button
          onClick={onClose}
          className="h-5 w-5 rounded-md hover:bg-destructive/10 flex items-center justify-center text-muted-foreground hover:text-destructive transition-colors"
        >
          <X className="h-3 w-3" />
        </button>
      </div>

      {/* Body */}
      {!minimized && (
        <div className="overflow-y-auto max-h-[calc(75vh-44px)] p-3 space-y-0">

          {/* Step 1: Embedding */}
          <PipelineStep
            icon={<Zap className="h-2 w-2" />}
            title="Query Embedding"
            timeMs={trace.embed_ms}
            active
          >
            <p className="text-[10px] text-muted-foreground">
              Vectorized query via embedding model
            </p>
          </PipelineStep>

          {/* Step 2: Retrieval */}
          <PipelineStep
            icon={<Search className="h-2 w-2" />}
            title={`${trace.search_method === 'hybrid' ? 'Hybrid' : 'Vector'} Search`}
            timeMs={trace.retrieval_ms}
            active
          >
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-[10px] text-muted-foreground">
                Fetched <span className="font-semibold text-foreground">{trace.candidates_fetched}</span> candidates
              </span>
              {trace.search_method === 'hybrid' && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium">
                  vector + keyword
                </span>
              )}
            </div>
            <div className="space-y-1">
              {trace.pre_rerank.map((chunk, i) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  chunk={chunk}
                  index={i + 1}
                  barColor="bg-blue-500"
                  maxScore={maxPreScore}
                />
              ))}
            </div>
          </PipelineStep>

          {/* Step 3: Reranking */}
          {trace.reranked ? (
            <PipelineStep
              icon={<ArrowRightLeft className="h-2 w-2" />}
              title="Cross-Encoder Reranking"
              timeMs={trace.rerank_ms}
              active
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-[10px] text-muted-foreground">
                  Reranked to <span className="font-semibold text-foreground">{trace.final_count}</span> results
                </span>
                {trace.rerank_model && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 font-mono truncate max-w-[160px]">
                    {trace.rerank_model}
                  </span>
                )}
              </div>
              <div className="space-y-1">
                {trace.post_rerank.map((chunk, i) => (
                  <ChunkCard
                    key={chunk.chunk_id}
                    chunk={chunk}
                    index={i + 1}
                    barColor="bg-emerald-500"
                    maxScore={maxPostScore}
                  />
                ))}
              </div>
            </PipelineStep>
          ) : (
            <PipelineStep
              icon={<ArrowDown className="h-2 w-2" />}
              title="Final Results"
              active
            >
              <span className="text-[10px] text-muted-foreground">
                <span className="font-semibold text-foreground">{trace.final_count}</span> chunks selected (reranker disabled)
              </span>
            </PipelineStep>
          )}

          {/* Step 4: Generation */}
          <PipelineStep
            icon={<FileText className="h-2 w-2" />}
            title="LLM Generation"
            active
          >
            <p className="text-[10px] text-muted-foreground">
              Context sent to LLM for answer synthesis
            </p>
          </PipelineStep>
        </div>
      )}
    </div>
  );
}
