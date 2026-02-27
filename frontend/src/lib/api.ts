// API Client for RAG Backend

import type {
  ChatRequest,
  ChatResponse,
  SearchRequest,
  SearchResponse,
  DocumentListResponse,
  HealthResponse,
  FileUploadResponse,
  IngestionRequest,
  IngestionResponse,
  Citation,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new APIError(
      errorData.message || errorData.error || 'API request failed',
      response.status,
      errorData
    );
  }
  return response.json();
}

export const api = {
  // Health Check
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_URL}/health`);
    return handleResponse(response);
  },

  // Chat (non-streaming)
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },

  // Chat (streaming)
  async chatStream(
    request: ChatRequest,
    onChunk: (chunk: string) => void,
    onComplete?: (fullResponse: string, citations: Citation[]) => void,
    onError?: (error: Error) => void,
    onCitations?: (citations: Citation[]) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new APIError('Stream request failed', response.status);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      let fullResponse = '';
      let citations: Citation[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              
              if (parsed.chunk) {
                fullResponse += parsed.chunk;
                onChunk(parsed.chunk);
              }
              
              if (parsed.citations) {
                citations = parsed.citations;
                onCitations?.(citations);
              }
              
              if (parsed.status === 'done') {
                onComplete?.(fullResponse, citations);
              }
              
              if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch (e) {
              // Ignore parse errors for non-JSON lines
            }
          }
        }
      }
    } catch (error) {
      onError?.(error as Error);
      throw error;
    }
  },

  // Search
  async search(request: SearchRequest): Promise<SearchResponse> {
    const response = await fetch(`${API_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },

  // Documents
  async getDocuments(limit = 100, offset = 0): Promise<DocumentListResponse> {
    const response = await fetch(
      `${API_URL}/documents?limit=${limit}&offset=${offset}`
    );
    return handleResponse(response);
  },

  // Upload File
  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  // Ingest Documents
  async ingestDocuments(request: IngestionRequest): Promise<IngestionResponse> {
    const response = await fetch(`${API_URL}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },
};

export { APIError };
