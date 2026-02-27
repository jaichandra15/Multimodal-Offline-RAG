'use client';

import { useQuery } from '@tanstack/react-query';
import { FileText, Database, Activity, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/lib/api';

export function Dashboard() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Documents</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {health?.knowledge_base.documents || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Total documents indexed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Chunks</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {health?.knowledge_base.chunks || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Total vector embeddings
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">
              {health?.status || 'Unknown'}
            </div>
            <p className="text-xs text-muted-foreground">System health</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Model Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">LLM Model:</span>
              <span className="font-medium">{health?.model_info.llm_model}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Embedding Model:</span>
              <span className="font-medium">
                {health?.model_info.embedding_model}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Dimensions:</span>
              <span className="font-medium">
                {health?.model_info.embedding_dimensions}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Service Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Database:</span>
              <span
                className={`font-medium ${
                  health?.database === 'connected'
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}
              >
                {health?.database}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Ollama:</span>
              <span
                className={`font-medium ${
                  health?.ollama === 'connected'
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}
              >
                {health?.ollama}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
