"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Activity, Clock, Database, Zap, TrendingUp, AlertCircle, ArrowLeft } from "lucide-react";

interface MetricsData {
  search_latency: { method: string; count: number; sum: number }[];
  generation_latency: { model: string; count: number; sum: number };
  chunks_retrieved: { count: number; sum: number };
  requests: { status: string; count: number }[];
  http_requests: { handler: string; method: string; status: string; count: number }[];
  ingestion: { documents: number; chunks: number };
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      const response = await fetch("http://localhost:8000/metrics");
      const text = await response.text();
      
      // Parse Prometheus metrics
      const parsed = parsePrometheusMetrics(text);
      setMetrics(parsed);
      setError(null);
    } catch (err) {
      setError("Failed to fetch metrics");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const parsePrometheusMetrics = (text: string): MetricsData => {
    const lines = text.split("\n");
    const data: MetricsData = {
      search_latency: [],
      generation_latency: { model: "", count: 0, sum: 0 },
      chunks_retrieved: { count: 0, sum: 0 },
      requests: [],
      http_requests: [],
      ingestion: { documents: 0, chunks: 0 },
    };

    for (const line of lines) {
      if (line.startsWith("#") || !line.trim()) continue;

      // Search latency
      if (line.includes("rag_search_latency_seconds_sum")) {
        const match = line.match(/method="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          const method = match[1];
          const sum = parseFloat(match[2]);
          const existing = data.search_latency.find(s => s.method === method);
          if (existing) {
            existing.sum = sum;
          } else {
            data.search_latency.push({ method, count: 0, sum });
          }
        }
      }
      if (line.includes("rag_search_latency_seconds_count")) {
        const match = line.match(/method="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          const method = match[1];
          const count = parseFloat(match[2]);
          const existing = data.search_latency.find(s => s.method === method);
          if (existing) {
            existing.count = count;
          }
        }
      }

      // Generation latency
      if (line.includes("rag_generation_latency_seconds_sum")) {
        const match = line.match(/model="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          data.generation_latency.model = match[1];
          data.generation_latency.sum = parseFloat(match[2]);
        }
      }
      if (line.includes("rag_generation_latency_seconds_count")) {
        const match = line.match(/model="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          data.generation_latency.count = parseFloat(match[2]);
        }
      }

      // Chunks retrieved
      if (line.includes("rag_chunks_retrieved_count_sum")) {
        const match = line.match(/([\d.]+)$/);
        if (match) data.chunks_retrieved.sum = parseFloat(match[1]);
      }
      if (line.includes("rag_chunks_retrieved_count_count")) {
        const match = line.match(/([\d.]+)$/);
        if (match) data.chunks_retrieved.count = parseFloat(match[1]);
      }

      // Request status
      if (line.includes("rag_requests_total{status=")) {
        const match = line.match(/status="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          data.requests.push({ status: match[1], count: parseFloat(match[2]) });
        }
      }

      // HTTP requests
      if (line.includes('http_requests_total{handler=')) {
        const match = line.match(/handler="([^"]+)",method="([^"]+)",status="([^"]+)"\}\s+([\d.]+)/);
        if (match) {
          data.http_requests.push({
            handler: match[1],
            method: match[2],
            status: match[3],
            count: parseFloat(match[4]),
          });
        }
      }

      // Ingestion
      if (line.includes("ingestion_chunks_created_total")) {
        const match = line.match(/([\d.]+)$/);
        if (match) data.ingestion.chunks = parseFloat(match[1]);
      }
    }

    return data;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="w-16 h-16 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-96">
          <CardHeader>
            <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-2" />
            <CardTitle className="text-center">Error Loading Metrics</CardTitle>
            <CardDescription className="text-center">{error}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  const avgSearchLatency = metrics.search_latency.reduce((acc, s) => acc + (s.count > 0 ? s.sum / s.count : 0), 0) / (metrics.search_latency.length || 1);
  const avgGenerationLatency = metrics.generation_latency.count > 0 ? metrics.generation_latency.sum / metrics.generation_latency.count : 0;
  const avgChunksRetrieved = metrics.chunks_retrieved.count > 0 ? metrics.chunks_retrieved.sum / metrics.chunks_retrieved.count : 0;
  const totalRequests = metrics.requests.reduce((acc, r) => acc + r.count, 0);

  const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'];

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Chat
            </Button>
          </Link>
          <div>
            <h1 className="text-4xl font-bold">System Metrics</h1>
            <p className="text-muted-foreground">Real-time RAG system performance monitoring</p>
          </div>
        </div>
        <Activity className="w-8 h-8 text-primary animate-pulse" />
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalRequests}</div>
            <p className="text-xs text-muted-foreground">
              {metrics.requests.find(r => r.status === "success")?.count || 0} successful
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Search Time</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgSearchLatency.toFixed(2)}s</div>
            <p className="text-xs text-muted-foreground">
              {metrics.search_latency[0]?.method || "hybrid"} search
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Generation</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgGenerationLatency.toFixed(2)}s</div>
            <p className="text-xs text-muted-foreground">
              {metrics.generation_latency.model} model
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Chunks</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgChunksRetrieved.toFixed(1)}</div>
            <p className="text-xs text-muted-foreground">per search</p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="requests">Requests</TabsTrigger>
          <TabsTrigger value="ingestion">Ingestion</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Search Latency */}
            <Card>
              <CardHeader>
                <CardTitle>Search Latency by Method</CardTitle>
                <CardDescription>Average time per search operation</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metrics.search_latency.map(s => ({
                    method: s.method,
                    avgLatency: s.count > 0 ? (s.sum / s.count) : 0,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="method" />
                    <YAxis label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Bar dataKey="avgLatency" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Generation Time */}
            <Card>
              <CardHeader>
                <CardTitle>LLM Generation Time</CardTitle>
                <CardDescription>Time spent generating responses</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={[
                    { name: 'Total', value: metrics.generation_latency.sum },
                    { name: 'Average', value: avgGenerationLatency },
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Area type="monotone" dataKey="value" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Chunks Retrieved Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Chunks Retrieved</CardTitle>
                <CardDescription>Number of chunks per request</CardDescription>
              </CardHeader>
              <CardContent className="h-80 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-6xl font-bold text-primary">{avgChunksRetrieved.toFixed(1)}</div>
                  <p className="text-sm text-muted-foreground mt-2">Average chunks per search</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Total: {metrics.chunks_retrieved.sum} chunks in {metrics.chunks_retrieved.count} searches
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Request Status */}
            <Card>
              <CardHeader>
                <CardTitle>Request Status</CardTitle>
                <CardDescription>Success vs error rate</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={metrics.requests}
                      dataKey="count"
                      nameKey="status"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {metrics.requests.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="requests" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>HTTP Requests by Endpoint</CardTitle>
              <CardDescription>Request count per API endpoint</CardDescription>
            </CardHeader>
            <CardContent className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metrics.http_requests}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="handler" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ingestion" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Ingestion Statistics</CardTitle>
              <CardDescription>Documents and chunks processed</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="text-center p-6 border rounded-lg">
                  <Database className="w-12 h-12 mx-auto mb-2 text-primary" />
                  <div className="text-3xl font-bold">{metrics.ingestion.documents}</div>
                  <p className="text-sm text-muted-foreground">Documents Processed</p>
                </div>
                <div className="text-center p-6 border rounded-lg">
                  <Activity className="w-12 h-12 mx-auto mb-2 text-primary" />
                  <div className="text-3xl font-bold">{metrics.ingestion.chunks}</div>
                  <p className="text-sm text-muted-foreground">Total Chunks Created</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
