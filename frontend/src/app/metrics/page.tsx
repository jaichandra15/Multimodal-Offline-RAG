"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import {
  Activity, Clock, Database, Zap, TrendingUp, AlertCircle,
  ArrowLeft, FileText, Layers, RefreshCw, CheckCircle2, XCircle,
  Server, Cpu, ShieldCheck, BarChart3,
} from "lucide-react";

// ── Constants ──────────────────────────────────────────────────────────────────
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://8000-01kjftqgyazawrg2zrdhsgqpp2.cloudspaces.litng.ai";

// Strip trailing slash to keep URLs clean
const BASE = API_URL.replace(/\/$/, "");

// ── Types ──────────────────────────────────────────────────────────────────────
interface PrometheusMetrics {
  search_latency: { method: string; count: number; sum: number }[];
  generation_latency: { model: string; count: number; sum: number };
  chunks_retrieved: { count: number; sum: number };
  requests: { status: string; count: number }[];
  http_requests: { handler: string; method: string; status: string; count: number }[];
  ingestion: { documents: number; chunks: number };
}

interface HealthData {
  status: string;
  database: string;
  ollama: string;
  knowledge_base: { documents: number; chunks: number };
  model_info: {
    llm_model: string;
    embedding_model: string;
    embedding_dimensions: number;
    hybrid_search: boolean;
    reranker_enabled: boolean;
    reranker_model: string | null;
  };
}

interface DocumentInfo {
  id: string;
  title: string;
  source: string;
  created_at: string;
}

// ── RAGAS types ────────────────────────────────────────────────────────────────
interface RAGASScores {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
}

interface RAGASEvaluation {
  id: string;
  question: string;
  answer: string;
  scores: RAGASScores;
  model_used: string;
  evaluated_at: string;
  has_reference: boolean;
}

interface RAGASHistory {
  evaluations: RAGASEvaluation[];
  total: number;
  averages: RAGASScores;
}

// ── Prometheus parser ──────────────────────────────────────────────────────────
function parsePrometheusMetrics(text: string): PrometheusMetrics {
  const lines = text.split("\n");
  const data: PrometheusMetrics = {
    search_latency: [],
    generation_latency: { model: "", count: 0, sum: 0 },
    chunks_retrieved: { count: 0, sum: 0 },
    requests: [],
    http_requests: [],
    ingestion: { documents: 0, chunks: 0 },
  };

  for (const line of lines) {
    if (line.startsWith("#") || !line.trim()) continue;

    if (line.includes("rag_search_latency_seconds_sum")) {
      const m = line.match(/method="([^"]+)"\}\s+([\d.]+)/);
      if (m) {
        const existing = data.search_latency.find((s) => s.method === m[1]);
        if (existing) existing.sum = parseFloat(m[2]);
        else data.search_latency.push({ method: m[1], count: 0, sum: parseFloat(m[2]) });
      }
    }
    if (line.includes("rag_search_latency_seconds_count")) {
      const m = line.match(/method="([^"]+)"\}\s+([\d.]+)/);
      if (m) {
        const existing = data.search_latency.find((s) => s.method === m[1]);
        if (existing) existing.count = parseFloat(m[2]);
      }
    }
    if (line.includes("rag_generation_latency_seconds_sum")) {
      const m = line.match(/model="([^"]+)"\}\s+([\d.]+)/);
      if (m) { data.generation_latency.model = m[1]; data.generation_latency.sum = parseFloat(m[2]); }
    }
    if (line.includes("rag_generation_latency_seconds_count")) {
      const m = line.match(/model="([^"]+)"\}\s+([\d.]+)/);
      if (m) data.generation_latency.count = parseFloat(m[2]);
    }
    if (line.includes("rag_chunks_retrieved_count_sum")) {
      const m = line.match(/([\d.]+)$/);
      if (m) data.chunks_retrieved.sum = parseFloat(m[1]);
    }
    if (line.includes("rag_chunks_retrieved_count_count")) {
      const m = line.match(/([\d.]+)$/);
      if (m) data.chunks_retrieved.count = parseFloat(m[1]);
    }
    if (line.includes('rag_requests_total{status=')) {
      const m = line.match(/status="([^"]+)"\}\s+([\d.]+)/);
      if (m) data.requests.push({ status: m[1], count: parseFloat(m[2]) });
    }
    if (line.includes('http_requests_total{handler=')) {
      const m = line.match(/handler="([^"]+)",method="([^"]+)",status="([^"]+)"\}\s+([\d.]+)/);
      if (m) data.http_requests.push({ handler: m[1], method: m[2], status: m[3], count: parseFloat(m[4]) });
    }
    if (line.includes("ingestion_chunks_created_total")) {
      const m = line.match(/([\d.]+)$/);
      if (m) data.ingestion.chunks = parseFloat(m[1]);
    }
  }
  return data;
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color = "violet",
}: {
  icon: any;
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  const colors: Record<string, string> = {
    violet: "from-violet-600 to-indigo-600 shadow-violet-500/20",
    emerald: "from-emerald-500 to-teal-600 shadow-emerald-500/20",
    amber:   "from-amber-500 to-orange-500 shadow-amber-500/20",
    sky:     "from-sky-500 to-blue-600 shadow-sky-500/20",
  };
  return (
    <Card className="border border-border/60 bg-card/60 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <div className={`bg-gradient-to-br ${colors[color]} p-2 rounded-xl shadow-lg`}>
          <Icon className="h-4 w-4 text-white" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold tracking-tight">{value}</div>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function MetricsPage() {
  const [prometheusMetrics, setPrometheusMetrics] = useState<PrometheusMetrics | null>(null);
  const [prometheusError, setPrometheusError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [ragasHistory, setRagasHistory] = useState<RAGASHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchAll = useCallback(async () => {
    setLoading(true);

    // 1. Health / knowledge base stats
    try {
      const res = await fetch(`${BASE}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: HealthData = await res.json();
      setHealth(data);
      setHealthError(null);
    } catch (e: any) {
      setHealthError(e.message || "Failed to fetch health data");
    }

    // 2. Documents list
    try {
      const res = await fetch(`${BASE}/documents?limit=100`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocuments(data.documents ?? []);
    } catch {
      setDocuments([]);
    }

    // 3. Prometheus metrics (best-effort — may not always be accessible)
    try {
      const res = await fetch(`${BASE}/metrics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      setPrometheusMetrics(parsePrometheusMetrics(text));
      setPrometheusError(null);
    } catch (e: any) {
      setPrometheusError("Prometheus metrics endpoint unavailable.");
      setPrometheusMetrics(null);
    }

    // 4. RAGAS evaluation history
    try {
      const res = await fetch(`${BASE}/evaluate/history?limit=50`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: RAGASHistory = await res.json();
      setRagasHistory(data);
    } catch {
      setRagasHistory(null);
    }

    setLastRefresh(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15_000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  // ── Derived values ───────────────────────────────────────────────────────────
  const totalDocs   = health?.knowledge_base.documents ?? 0;
  const totalChunks = health?.knowledge_base.chunks    ?? 0;
  const avgChunksPerDoc = totalDocs > 0 ? (totalChunks / totalDocs).toFixed(1) : "0";

  // null = no Prometheus data yet (distinct from a real 0 measurement)
  const avgSearchLatency: number | null = prometheusMetrics
    ? (() => {
        const withData = prometheusMetrics.search_latency.filter((s) => s.count > 0);
        if (withData.length === 0) return null;
        return withData.reduce((acc, s) => acc + s.sum / s.count, 0) / withData.length;
      })()
    : null;

  const avgGenerationLatency: number | null = prometheusMetrics
    ? (prometheusMetrics.generation_latency.count > 0
        ? prometheusMetrics.generation_latency.sum / prometheusMetrics.generation_latency.count
        : null)
    : null;

  const totalRequests = prometheusMetrics
    ? prometheusMetrics.requests.reduce((acc, r) => acc + r.count, 0)
    : null;

  const COLORS = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981"];

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex items-center gap-4 px-6 h-14">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Chat
            </Button>
          </Link>
          <div className="h-5 w-px bg-border" />
          <div className="flex-1">
            <h1 className="text-sm font-semibold">System Dashboard</h1>
            <p className="text-[11px] text-muted-foreground">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2 h-8 text-xs"
            onClick={fetchAll}
            disabled={loading}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          {/* System status pill */}
          {health && (
            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
                health.status === "healthy"
                  ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/30"
                  : "bg-amber-500/10 text-amber-600 border-amber-500/30"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  health.status === "healthy" ? "bg-emerald-500" : "bg-amber-500"
                } animate-pulse`}
              />
              {health.status === "healthy" ? "All Systems Operational" : "Degraded"}
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Health error banner */}
        {healthError && (
          <div className="flex items-center gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>Could not reach backend: <strong>{healthError}</strong>. Make sure the backend is running.</span>
          </div>
        )}

        {/* Top stat cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            icon={FileText}
            label="Total Documents"
            value={totalDocs}
            sub="in knowledge base"
            color="violet"
          />
          <StatCard
            icon={Layers}
            label="Total Embeddings"
            value={totalChunks}
            sub={`~${avgChunksPerDoc} chunks per doc`}
            color="emerald"
          />
          <StatCard
            icon={TrendingUp}
            label="Total Requests"
            value={totalRequests ?? "N/A"}
            sub={
              totalRequests != null
                ? `${prometheusMetrics?.requests.find((r) => r.status === "success")?.count ?? 0} successful`
                : "no data yet"
            }
            color="sky"
          />
          <StatCard
            icon={Zap}
            label="Avg Search Time"
            value={avgSearchLatency != null ? `${avgSearchLatency.toFixed(2)}s` : "N/A"}
            sub={avgSearchLatency != null ? "per query" : "send a query to measure"}
            color="amber"
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="knowledge" className="space-y-4">
          <TabsList className="h-9">
            <TabsTrigger value="knowledge" className="text-xs gap-1.5">
              <Database className="h-3.5 w-3.5" />
              Knowledge Base
            </TabsTrigger>
            <TabsTrigger value="performance" className="text-xs gap-1.5">
              <Activity className="h-3.5 w-3.5" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="ragas" className="text-xs gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5" />
              RAGAS Quality
            </TabsTrigger>
            <TabsTrigger value="system" className="text-xs gap-1.5">
              <Server className="h-3.5 w-3.5" />
              System Info
            </TabsTrigger>
          </TabsList>

          {/* ── Knowledge Base Tab ─────────────────────────────── */}
          <TabsContent value="knowledge" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              {/* Docs vs Chunks visual */}
              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle className="text-sm font-semibold">Storage Overview</CardTitle>
                  <CardDescription className="text-xs">Docs & embeddings breakdown</CardDescription>
                </CardHeader>
                <CardContent className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: "Documents", value: totalDocs },
                        { name: "Embeddings", value: totalChunks },
                      ]}
                      barSize={40}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                        {[COLORS[0], COLORS[1]].map((c, i) => (
                          <Cell key={i} fill={c} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Document list */}
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="text-sm font-semibold">
                    Indexed Documents
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      ({documents.length} total)
                    </span>
                  </CardTitle>
                  <CardDescription className="text-xs">All files in the knowledge base</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-52">
                    {documents.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-40 text-center gap-2">
                        <Database className="h-8 w-8 text-muted-foreground/40" />
                        <p className="text-sm text-muted-foreground">No documents yet</p>
                        <p className="text-xs text-muted-foreground/60">
                          Upload files using the 📎 button in the chat
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2 pr-2">
                        {documents.map((doc) => (
                          <div
                            key={doc.id}
                            className="flex items-center gap-3 p-2.5 rounded-lg border border-border/50 bg-muted/30"
                          >
                            <div className="h-8 w-8 rounded-lg bg-violet-500/10 flex items-center justify-center flex-shrink-0">
                              <FileText className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-medium truncate">{doc.title}</p>
                              <p className="text-[10px] text-muted-foreground truncate">{doc.source}</p>
                            </div>
                            <p className="text-[10px] text-muted-foreground flex-shrink-0">
                              {doc.created_at
                                ? new Date(doc.created_at).toLocaleDateString()
                                : "—"}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ── Performance Tab ────────────────────────────────── */}
          <TabsContent value="performance" className="space-y-4">
            {prometheusError && (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-sm text-amber-700 dark:text-amber-400">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <span>{prometheusError} Performance charts will appear after the first chat query.</span>
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-2">
              {/* Search Latency */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Search Latency by Method</CardTitle>
                  <CardDescription className="text-xs">Average seconds per operation</CardDescription>
                </CardHeader>
                <CardContent className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={
                        (prometheusMetrics?.search_latency ?? []).map((s) => ({
                          method: s.method,
                          avgLatency: s.count > 0 ? +(s.sum / s.count).toFixed(3) : 0,
                        }))
                      }
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="method" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} unit="s" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
                      />
                      <Bar dataKey="avgLatency" fill={COLORS[0]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Request Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Request Status</CardTitle>
                  <CardDescription className="text-xs">Success vs error rate</CardDescription>
                </CardHeader>
                <CardContent className="h-64">
                  {(prometheusMetrics?.requests ?? []).length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={prometheusMetrics!.requests}
                          dataKey="count"
                          nameKey="status"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          label={({ status, percent }) =>
                            `${status} ${(percent * 100).toFixed(0)}%`
                          }
                        >
                          {prometheusMetrics!.requests.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
                      No request data yet — send a chat message first
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Generation time */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">LLM Generation Time</CardTitle>
                  <CardDescription className="text-xs">Total & average seconds for LLM responses</CardDescription>
                </CardHeader>
                <CardContent className="h-64">
                  {avgGenerationLatency != null ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={[
                          { name: "Total", value: prometheusMetrics?.generation_latency.sum ?? 0 },
                          { name: "Average", value: +avgGenerationLatency.toFixed(3) },
                        ]}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} unit="s" />
                        <Tooltip
                          contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }}
                        />
                        <Area type="monotone" dataKey="value" stroke={COLORS[1]} fill={COLORS[1]} fillOpacity={0.25} />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
                      No generation data yet — send a chat message first
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Chunks retrieved */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Chunks Retrieved</CardTitle>
                  <CardDescription className="text-xs">Average chunks returned per search</CardDescription>
                </CardHeader>
                <CardContent className="h-64 flex flex-col items-center justify-center gap-2">
                  <div className="text-7xl font-bold bg-gradient-to-br from-violet-600 to-indigo-600 bg-clip-text text-transparent">
                    {prometheusMetrics && prometheusMetrics.chunks_retrieved.count > 0
                      ? (prometheusMetrics.chunks_retrieved.sum / prometheusMetrics.chunks_retrieved.count).toFixed(1)
                      : "N/A"}
                  </div>
                  <p className="text-sm text-muted-foreground">avg chunks per search</p>
                  {prometheusMetrics && (
                    <p className="text-xs text-muted-foreground">
                      {prometheusMetrics.chunks_retrieved.sum} total chunks ·{" "}
                      {prometheusMetrics.chunks_retrieved.count} searches
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ── System Info Tab ─────────────────────────────────── */}
          <TabsContent value="system" className="space-y-4">
            {health ? (
              <div className="grid gap-4 md:grid-cols-2">
                {/* Services */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Service Health</CardTitle>
                    <CardDescription className="text-xs">Current status of backend services</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      { label: "Database (PostgreSQL)", value: health.database },
                      { label: "Ollama LLM Service", value: health.ollama },
                      { label: "RAG Backend", value: health.status },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                        <span className="text-sm text-muted-foreground">{label}</span>
                        <div className={`flex items-center gap-1.5 text-xs font-medium ${
                          value === "connected" || value === "healthy"
                            ? "text-emerald-600"
                            : "text-destructive"
                        }`}>
                          {value === "connected" || value === "healthy"
                            ? <CheckCircle2 className="h-3.5 w-3.5" />
                            : <XCircle className="h-3.5 w-3.5" />}
                          {value}
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                {/* Model info */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Model Configuration</CardTitle>
                    <CardDescription className="text-xs">Active Ollama models & settings</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      { label: "LLM Model", value: health.model_info.llm_model },
                      { label: "Embedding Model", value: health.model_info.embedding_model },
                      { label: "Embedding Dimensions", value: String(health.model_info.embedding_dimensions) },
                      { label: "Hybrid Search", value: health.model_info.hybrid_search ? "Enabled" : "Disabled" },
                      {
                        label: "Reranker",
                        value: health.model_info.reranker_enabled
                          ? `Enabled (${health.model_info.reranker_model})`
                          : "Disabled",
                      },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                        <span className="text-sm text-muted-foreground">{label}</span>
                        <span className="text-xs font-medium font-mono bg-muted px-2 py-0.5 rounded">
                          {value}
                        </span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="flex items-center justify-center h-40 text-muted-foreground text-sm gap-2">
                <AlertCircle className="h-5 w-5" />
                Unable to load system info — backend may be unreachable.
              </div>
            )}
          </TabsContent>

          {/* ── RAGAS Quality tab ──────────────────────────────────────────── */}
          <TabsContent value="ragas" className="space-y-4">
            <div className="flex items-start gap-3 p-4 rounded-xl bg-violet-500/10 border border-violet-500/20 text-sm">
              <ShieldCheck className="h-4 w-4 text-violet-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-violet-600 dark:text-violet-400">RAGAS Auto-Evaluation</p>
                <p className="text-muted-foreground text-xs mt-0.5">
                  Scores are computed automatically in the background after each chat response using Ollama as judge. Allow 15–60 s per query.
                </p>
              </div>
            </div>

            {ragasHistory && ragasHistory.total > 0 ? (
              <>
                {/* 4 score cards */}
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {([
                    { label: "Faithfulness",      value: ragasHistory.averages.faithfulness,      desc: "Answer grounded in context" },
                    { label: "Answer Relevancy",  value: ragasHistory.averages.answer_relevancy,  desc: "Answer addresses question" },
                    { label: "Context Precision", value: ragasHistory.averages.context_precision, desc: "Needs ground truth" },
                    { label: "Context Recall",    value: ragasHistory.averages.context_recall,    desc: "Needs ground truth" },
                  ]).map(({ label, value, desc }) => {
                    const pct = value != null ? Math.round(value * 100) : null;
                    const quality = pct == null ? "nodata" : pct >= 75 ? "good" : pct >= 50 ? "ok" : "poor";
                    const bar  = { good: "bg-emerald-500", ok: "bg-amber-500", poor: "bg-red-500", nodata: "bg-muted-foreground/20" }[quality];
                    const txt  = { good: "text-emerald-500", ok: "text-amber-500", poor: "text-red-500", nodata: "text-muted-foreground" }[quality];
                    return (
                      <Card key={label} className="border border-border/60 bg-card/60 backdrop-blur">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="text-3xl font-bold tracking-tight">{pct != null ? `${pct}%` : "N/A"}</div>
                          <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div className={`h-full rounded-full transition-all ${bar}`} style={{ width: pct != null ? `${pct}%` : "0%" }} />
                          </div>
                          <p className={`text-xs mt-1.5 font-medium ${txt}`}>
                            {quality === "nodata" ? desc : `${quality.toUpperCase()} • ${desc}`}
                          </p>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>

                {/* Score trend */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Score Trend</CardTitle>
                    <CardDescription className="text-xs">Faithfulness &amp; Answer Relevancy over last {ragasHistory.total} evals</CardDescription>
                  </CardHeader>
                  <CardContent className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={[...ragasHistory.evaluations].reverse().map((e, i) => ({
                        idx: i + 1,
                        faith: e.scores.faithfulness != null ? +(e.scores.faithfulness * 100).toFixed(1) : null,
                        rel:   e.scores.answer_relevancy != null ? +(e.scores.answer_relevancy * 100).toFixed(1) : null,
                        label: new Date(e.evaluated_at).toLocaleTimeString(),
                      }))}>
                        <defs>
                          <linearGradient id="gF2" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="#8b5cf6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="gA2" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="label" tick={{ fontSize: 10 }} tickLine={false} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                        <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} formatter={(v: any) => [`${v}%`]} />
                        <Legend />
                        <Area type="monotone" dataKey="faith" name="Faithfulness"    stroke="#8b5cf6" fill="url(#gF2)" connectNulls />
                        <Area type="monotone" dataKey="rel"   name="Answer Relevancy" stroke="#06b6d4" fill="url(#gA2)" connectNulls />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Query table */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Evaluated Queries</CardTitle>
                    <CardDescription className="text-xs">Most recent {ragasHistory.total} (newest first)</CardDescription>
                  </CardHeader>
                  <CardContent className="p-0">
                    <ScrollArea className="h-80">
                      <table className="w-full text-xs">
                        <thead className="sticky top-0 bg-muted/70 backdrop-blur">
                          <tr>
                            <th className="text-left px-4 py-2 font-medium text-muted-foreground">Question</th>
                            <th className="text-center px-3 py-2 font-medium text-muted-foreground">Faith.</th>
                            <th className="text-center px-3 py-2 font-medium text-muted-foreground">Relevancy</th>
                            <th className="text-center px-3 py-2 font-medium text-muted-foreground">Prec.</th>
                            <th className="text-center px-3 py-2 font-medium text-muted-foreground">Recall</th>
                            <th className="text-right px-4 py-2 font-medium text-muted-foreground">Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ragasHistory.evaluations.map((ev, i) => {
                            const fmt = (v: number | null) =>
                              v != null
                                ? <span className={`font-mono font-semibold ${v >= 0.75 ? "text-emerald-500" : v >= 0.5 ? "text-amber-500" : "text-red-500"}`}>{(v * 100).toFixed(0)}%</span>
                                : <span className="text-muted-foreground/40">—</span>;
                            return (
                              <tr key={ev.id} className={`border-t border-border/40 ${i % 2 === 0 ? "bg-background" : "bg-muted/20"}`}>
                                <td className="px-4 py-2.5 max-w-[220px]">
                                  <p className="truncate text-muted-foreground" title={ev.question}>{ev.question}</p>
                                </td>
                                <td className="text-center px-3 py-2.5">{fmt(ev.scores.faithfulness)}</td>
                                <td className="text-center px-3 py-2.5">{fmt(ev.scores.answer_relevancy)}</td>
                                <td className="text-center px-3 py-2.5">{fmt(ev.scores.context_precision)}</td>
                                <td className="text-center px-3 py-2.5">{fmt(ev.scores.context_recall)}</td>
                                <td className="text-right px-4 py-2.5 text-muted-foreground whitespace-nowrap">
                                  {new Date(ev.evaluated_at).toLocaleTimeString()}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center h-48 gap-3 text-center">
                  <ShieldCheck className="h-10 w-10 text-muted-foreground/30" />
                  <div>
                    <p className="text-sm font-medium">No evaluations yet</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Send a few chat messages — RAGAS scores each one automatically in the background.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
