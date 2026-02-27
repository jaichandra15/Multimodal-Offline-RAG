
export interface Citation {
  number: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  document_source: string;
  content: string;
  metadata: Record<string, any>;
  similarity?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export interface ChatRequest {
  message: string;
  conversation_history?: ChatMessage[];
}

export interface ChatResponse {
  response: string;
  conversation_history: ChatMessage[];
  citations: Citation[];
}

export interface SearchRequest {
  query: string;
  limit?: number;
}

export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  content: string;
  similarity: number;
  metadata: Record<string, any>;
  document_title: string;
  document_source: string;
}

export interface SearchResponse {
  results: SearchResultItem[];
  total_results: number;
}

export interface DocumentInfo {
  id: string;
  title: string;
  source: string;
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
  chunk_count?: number;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface HealthResponse {
  status: string;
  database: string;
  ollama: string;
  knowledge_base: {
    documents: number;
    chunks: number;
  };
  model_info: {
    llm_model: string;
    embedding_model: string;
    embedding_dimensions: number;
  };
}

export interface FileUploadResponse {
  status: string;
  message: string;
  document_id?: string;
  chunks_created: number;
  filename: string;
}

export interface IngestionRequest {
  clean_existing?: boolean;
  documents_path?: string;
}

export interface IngestionResponse {
  status: string;
  message: string;
  documents_processed: number;
  chunks_created: number;
  errors: string[];
}
