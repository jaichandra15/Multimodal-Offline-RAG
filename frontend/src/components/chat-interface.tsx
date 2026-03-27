'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send,
  Loader2,
  Paperclip,
  FileText,
  X,
  Sparkles,
  MessageSquare,
  Trash2,
  Plus,
  BookOpen,
  ChevronDown,
  ChevronUp,
  User,
  Edit2,
  Check,
  BarChart3,
  Database,
  Upload,
  CheckCircle2,
  AlertCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Bot,
  Layers,
} from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useChatStore, useSettingsStore } from '@/lib/store';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useQuery } from '@tanstack/react-query';
import { formatDistanceToNow } from 'date-fns';
import type { Citation, RetrievalTrace } from '@/lib/types';
import { RetrievalPipelinePanel } from '@/components/retrieval-panel';

// ─── Typing Indicator ────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex gap-4 justify-start">
      <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-2 rounded-xl h-9 w-9 flex-shrink-0 flex items-center justify-center shadow-md">
        <Bot className="h-4 w-4 text-white" />
      </div>
      <div className="rounded-2xl px-4 py-3 bg-card border border-border shadow-sm flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:0ms]" />
        <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:150ms]" />
        <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:300ms]" />
      </div>
    </div>
  );
}

// ─── Citation Panel ───────────────────────────────────────────────────────────
function CitationPanel({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-2 rounded-xl border border-border bg-muted/40 overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
      >
        <BookOpen className="h-3.5 w-3.5 flex-shrink-0" />
        <span className="font-medium">
          {citations.length} source{citations.length > 1 ? 's' : ''} cited
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 ml-auto" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 ml-auto" />
        )}
      </button>

      {expanded && (
        <div className="px-3 pb-3 grid gap-1.5">
          {citations.map((c) => {
            const filename = c.document_source.split('/').pop() || c.document_source;
            const pageNum = c.metadata?.page || c.metadata?.page_number;
            return (
              <div
                key={c.number}
                className="flex items-start gap-2.5 p-2 rounded-lg bg-background border border-border/60"
              >
                <span className="flex-shrink-0 h-5 w-5 rounded-md bg-violet-500/10 text-violet-600 dark:text-violet-400 flex items-center justify-center text-[10px] font-bold mt-0.5">
                  {c.number}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{c.document_title || filename}</p>
                  <p className="text-[11px] text-muted-foreground truncate">
                    {filename}{pageNum ? ` · Page ${pageNum}` : ''}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Document Panel (overlay modal) ─────────────────────────────────────────
function DocumentPanel({
  open,
  onClose,
  documents,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  documents: any[];
  loading: boolean;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onClose} />
      {/* Panel */}
      <div className="relative z-10 flex flex-col h-full w-80 bg-background border-l shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-violet-500" />
            <h3 className="font-semibold text-sm">Knowledge Base</h3>
            {documents.length > 0 && (
              <span className="h-5 min-w-[20px] px-1.5 rounded-full bg-muted text-[10px] font-medium flex items-center justify-center">
                {documents.length}
              </span>
            )}
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <ScrollArea className="flex-1 p-3">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-center gap-3">
              <Upload className="h-8 w-8 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No documents yet</p>
              <p className="text-xs text-muted-foreground/70">
                Upload a file using the paperclip icon in the chat input
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc: any) => (
                <Card key={doc.id} className="p-3">
                  <div className="flex items-start gap-2.5">
                    <div className="h-8 w-8 rounded-lg bg-violet-500/10 flex items-center justify-center flex-shrink-0">
                      <FileText className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate leading-tight">{doc.title}</p>
                      <p className="text-xs text-muted-foreground truncate mt-0.5">{doc.source}</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}

// ─── Upload Progress Banner ──────────────────────────────────────────────────
function UploadBanner({
  uploading,
  progress,
  filename,
}: {
  uploading: boolean;
  progress: number;
  filename: string;
}) {
  if (!uploading) return null;
  return (
    <div className="mx-4 mb-3 p-3 bg-violet-500/10 border border-violet-500/30 rounded-xl flex items-center gap-3">
      <div className="h-8 w-8 rounded-lg bg-violet-500/20 flex items-center justify-center flex-shrink-0">
        <Upload className="h-4 w-4 text-violet-600 dark:text-violet-400 animate-pulse" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate text-violet-700 dark:text-violet-300">{filename}</p>
        <div className="flex items-center gap-2 mt-1">
          <Progress value={progress} className="h-1.5 flex-1" />
          <span className="text-xs text-violet-600 dark:text-violet-400 font-medium">
            {progress}%
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Empty State ─────────────────────────────────────────────────────────────
const STARTER_PROMPTS = [
  'Summarize the key points from the uploaded documents',
  'What topics are covered in the knowledge base?',
  'Find information about a specific topic',
];

function EmptyState({ onPrompt }: { onPrompt: (s: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center px-4">
      <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-5 rounded-3xl mb-5 shadow-lg shadow-violet-500/20">
        <Sparkles className="h-12 w-12 text-white" />
      </div>
      <h2 className="text-2xl font-bold mb-2">RAG Knowledge Assistant</h2>
      <p className="text-muted-foreground max-w-sm mb-8 text-sm leading-relaxed">
        Upload documents using the <strong>📎 paperclip</strong> button, then ask questions. I'll
        find answers from your knowledge base.
      </p>

      <div className="w-full max-w-sm space-y-2">
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-3">
          Try asking
        </p>
        {STARTER_PROMPTS.map((p) => (
          <button
            key={p}
            onClick={() => onPrompt(p)}
            className="w-full text-left px-4 py-2.5 rounded-xl border border-border bg-card hover:bg-muted hover:border-violet-400/50 transition-all text-sm text-muted-foreground hover:text-foreground"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export function ChatInterface() {
  const [input, setInput] = useState('');
  const [streamingText, setStreamingText] = useState('');
  const [streamingCitations, setStreamingCitations] = useState<Citation[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadingFileName, setUploadingFileName] = useState('');
  const [showDocs, setShowDocs] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [retrievalTrace, setRetrievalTrace] = useState<RetrievalTrace | null>(null);
  const [showPipeline, setShowPipeline] = useState(false);

  const {
    conversations,
    currentConversationId,
    createConversation,
    deleteConversation,
    setCurrentConversation,
    renameConversation,
    addMessage,
    setIsStreaming,
    isStreaming,
    getCurrentMessages,
  } = useChatStore();

  const messages = getCurrentMessages();
  const { useStreaming } = useSettingsStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch documents for the knowledge base panel
  const { data: docsData, isLoading: docsLoading, refetch: refetchDocs } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(100),
    enabled: showDocs,
  });

  const { data: docCountData } = useQuery({
    queryKey: ['documents-count'],
    queryFn: () => api.getDocuments(1),
    refetchInterval: 30_000,
  });

  const docCount = docCountData?.total ?? 0;

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  // ── File Upload ────────────────────────────────────────────────────────────
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadingFileName(file.name);

    addMessage({
      role: 'assistant',
      content: `📎 Processing **${file.name}**…`,
    });

    const interval = setInterval(() => {
      setUploadProgress((p) => Math.min(p + 8, 88));
    }, 250);

    try {
      const resp = await api.uploadFile(file);
      setUploadProgress(100);
      addMessage({
        role: 'assistant',
        content: `✅ **${file.name}** uploaded successfully!\n\nCreated **${resp.chunks_created}** searchable chunks. You can now ask questions about this document.`,
      });
      toast.success('Document added to knowledge base');
      refetchDocs();
    } catch (err: any) {
      addMessage({
        role: 'assistant',
        content: `❌ Failed to upload **${file.name}**: ${err.message}`,
      });
      toast.error(`Upload failed: ${err.message}`);
    } finally {
      clearInterval(interval);
      setUploading(false);
      setUploadProgress(0);
      setUploadingFileName('');
      e.target.value = '';
    }
  };

  // ── Send Message ───────────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;
  
    const userMsg = { role: 'user' as const, content: text };
    addMessage(userMsg);
    setInput('');
    setIsStreaming(true);
  
    const timeout = setTimeout(() => {
      console.warn("Force reset streaming");
      setIsStreaming(false);
    }, 50000); // 30s safety
  
    try {
      if (useStreaming) {
        setStreamingText('');
        setStreamingCitations([]);
        setRetrievalTrace(null);
  
        await api.chatStream(
          { message: userMsg.content, conversation_history: messages },
          (chunk) => setStreamingText((prev) => prev + chunk),
          (fullResponse, citations) => {
            addMessage({ role: 'assistant', content: fullResponse, citations });
            setStreamingText('');
            setStreamingCitations([]);
            setIsStreaming(false);
          },
          (error) => {
            toast.error(`Error: ${error.message}`);
            setStreamingText('');
            setStreamingCitations([]);
            setIsStreaming(false);
          },
          (citations) => setStreamingCitations(citations),
          (trace) => {
            setRetrievalTrace(trace);
            setShowPipeline(true);
          },
        );
      } else {
        const resp = await api.chat({ message: userMsg.content, conversation_history: messages });
        addMessage({ role: 'assistant', content: resp.response, citations: resp.citations });
      }
    } catch (err) {
      console.error(err);
    } finally {
      clearTimeout(timeout);
      setIsStreaming(false);   // 🔥 GUARANTEED RESET
    }
  }, [input, isStreaming, messages, useStreaming, addMessage, setIsStreaming]);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* ─── Conversation Sidebar ─────────────────────────────────────────── */}
      {showSidebar && (
        <aside className="w-60 border-r bg-muted/20 flex flex-col h-full flex-shrink-0">
          {/* Sidebar header */}
          <div className="p-3 border-b flex items-center gap-2 flex-shrink-0">
            <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-1.5 rounded-lg">
              <Sparkles className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-sm font-semibold flex-1">RAG Assistant</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setShowSidebar(false)}
              title="Collapse sidebar"
            >
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </div>

          {/* New chat button */}
          <div className="p-2 border-b flex-shrink-0">
            <Button
              onClick={() => createConversation()}
              className="w-full justify-start gap-2 h-8 text-sm"
              variant="outline"
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </Button>
          </div>

          {/* Conversation list */}
          <ScrollArea className="flex-1 overflow-y-auto">
            <div className="p-2 space-y-0.5">
              {conversations.length === 0 && (
                <div className="text-center py-8 text-xs text-muted-foreground">
                  No conversations yet
                </div>
              )}
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`group flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-colors ${
                    currentConversationId === conv.id
                      ? 'bg-violet-500/10 text-violet-700 dark:text-violet-300'
                      : 'hover:bg-muted/60 text-muted-foreground hover:text-foreground'
                  }`}
                  onClick={() => {
                    if (editingId !== conv.id) setCurrentConversation(conv.id);
                  }}
                >
                  <MessageSquare className="h-3.5 w-3.5 flex-shrink-0" />

                  {editingId === conv.id ? (
                    <input
                      type="text"
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          renameConversation(conv.id, editingTitle);
                          setEditingId(null);
                        } else if (e.key === 'Escape') {
                          setEditingId(null);
                        }
                      }}
                      className="flex-1 text-xs bg-background border border-border rounded px-1.5 py-0.5 outline-none"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{conv.title}</p>
                      <p className="text-[10px] text-muted-foreground/70">
                        {formatDistanceToNow(conv.updatedAt, { addSuffix: true })}
                      </p>
                    </div>
                  )}

                  {editingId === conv.id ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5 flex-shrink-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        renameConversation(conv.id, editingTitle);
                        setEditingId(null);
                      }}
                    >
                      <Check className="h-3 w-3" />
                    </Button>
                  ) : (
                    <div className="hidden group-hover:flex items-center gap-0.5">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-5 w-5"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingId(conv.id);
                          setEditingTitle(conv.title);
                        }}
                      >
                        <Edit2 className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-5 w-5 text-destructive/70 hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteConversation(conv.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>

          {/* Sidebar footer */}
          <div className="p-2 border-t flex-shrink-0">
            <Link href="/metrics">
              <Button variant="ghost" className="w-full justify-start gap-2 h-8 text-xs text-muted-foreground">
                <BarChart3 className="h-3.5 w-3.5" />
                Metrics Dashboard
              </Button>
            </Link>
          </div>
        </aside>
      )}

      {/* ─── Main Chat Panel ─────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* Top bar */}
        <header className="border-b bg-background/95 backdrop-blur flex-shrink-0">
          <div className="flex items-center gap-3 px-4 h-14">
            {/* Collapse/expand sidebar */}
            {!showSidebar && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setShowSidebar(true)}
                title="Open sidebar"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
            )}

            {/* Current conversation title (or welcome) */}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold truncate">
                {conversations.find((c) => c.id === currentConversationId)?.title ?? 'New Chat'}
              </p>
              <p className="text-[11px] text-muted-foreground">Powered by Ollama</p>
            </div>

            {/* Pipeline trace button */}
            {retrievalTrace && (
              <Button
                variant={showPipeline ? 'default' : 'outline'}
                size="sm"
                className={`gap-2 h-8 text-xs ${
                  showPipeline
                    ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white border-0'
                    : ''
                }`}
                onClick={() => setShowPipeline((v) => !v)}
              >
                <Layers className="h-3.5 w-3.5" />
                Pipeline
              </Button>
            )}

            {/* Knowledge base button */}
            <Button
              variant="outline"
              size="sm"
              className="gap-2 h-8 text-xs"
              onClick={() => setShowDocs(true)}
            >
              <Database className="h-3.5 w-3.5" />
              {docCount > 0 ? (
                <span>
                  {docCount} Doc{docCount !== 1 ? 's' : ''}
                </span>
              ) : (
                <span>Knowledge Base</span>
              )}
            </Button>

            {/* New chat */}
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-xs gap-1.5"
              onClick={() => createConversation()}
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </Button>
          </div>
        </header>

        {/* Message list */}
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6">
            {messages.length === 0 && !streamingText ? (
              <EmptyState onPrompt={(s) => setInput(s)} />
            ) : (
              <div className="space-y-6">
                {messages.map((msg, idx) => (
                  <div key={idx}>
                    <div
                      className={`flex gap-3 ${
                        msg.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      {/* Bot avatar */}
                      {msg.role === 'assistant' && (
                        <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-2 rounded-xl h-9 w-9 flex-shrink-0 flex items-center justify-center shadow-md">
                          <Bot className="h-4 w-4 text-white" />
                        </div>
                      )}

                      {/* Bubble */}
                      <div
                        className={
                          msg.role === 'user' ? 'max-w-[78%]' : 'flex-1 max-w-[84%]'
                        }
                      >
                        <div
                          className={`rounded-2xl px-4 py-3 shadow-sm ${
                            msg.role === 'user'
                              ? 'bg-gradient-to-br from-violet-600 to-indigo-600 text-white'
                              : 'bg-card border border-border'
                          }`}
                        >
                          <div
                            className={`prose prose-sm max-w-none ${
                              msg.role === 'user'
                                ? 'prose-invert'
                                : 'dark:prose-invert'
                            }`}
                          >
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code({ node, inline, className, children, ...props }: any) {
                                  const match = /language-(\w+)/.exec(className || '');
                                  return !inline && match ? (
                                    <SyntaxHighlighter
                                      style={vscDarkPlus}
                                      language={match[1]}
                                      PreTag="div"
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, '')}
                                    </SyntaxHighlighter>
                                  ) : (
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                              }}
                            >
                              {msg.content}
                            </ReactMarkdown>
                          </div>
                        </div>

                        {/* Citations */}
                        {msg.role === 'assistant' && msg.citations && (
                          <CitationPanel citations={msg.citations} />
                        )}
                      </div>

                      {/* User avatar */}
                      {msg.role === 'user' && (
                        <div className="bg-muted border border-border p-2 rounded-xl h-9 w-9 flex-shrink-0 flex items-center justify-center">
                          <User className="h-4 w-4 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Streaming response */}
                {isStreaming && (
                  <div>
                    {streamingText ? (
                      <div className="flex gap-3 justify-start">
                        <div className="bg-gradient-to-br from-violet-600 to-indigo-600 p-2 rounded-xl h-9 w-9 flex-shrink-0 flex items-center justify-center shadow-md">
                          <Bot className="h-4 w-4 text-white" />
                        </div>
                        <div className="flex-1 max-w-[84%]">
                          <div className="rounded-2xl px-4 py-3 bg-card border border-border shadow-sm">
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {streamingText}
                              </ReactMarkdown>
                            </div>
                          </div>
                          {streamingCitations.length > 0 && (
                            <CitationPanel citations={streamingCitations} />
                          )}
                        </div>
                      </div>
                    ) : (
                      <TypingIndicator />
                    )}
                  </div>
                )}

                <div ref={scrollRef} />
              </div>
            )}
          </div>
        </ScrollArea>

        {/* ── Input area ──────────────────────────────────────────────────── */}
        <div className="border-t bg-background/95 backdrop-blur flex-shrink-0">
          {/* Upload progress */}
          <UploadBanner
            uploading={uploading}
            progress={uploadProgress}
            filename={uploadingFileName}
          />

          <div className="max-w-3xl mx-auto px-4 py-3">
            <div className="flex gap-2 items-end">
              {/* File attach */}
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleFileUpload}
                disabled={uploading || isStreaming}
                accept=".pdf,.docx,.pptx,.xlsx,.md,.txt,.mp3,.wav,.m4a,.flac"
              />
              <Button
                variant="outline"
                size="icon"
                className="h-10 w-10 flex-shrink-0 rounded-xl"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || isStreaming}
                title="Upload document"
              >
                <Paperclip className="h-4 w-4" />
              </Button>

              {/* Message input */}
              <div className="flex-1 relative">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question… (Enter to send, Shift+Enter for new line)"
                  className="min-h-[42px] max-h-[160px] resize-none rounded-xl py-2.5 pr-3 text-sm leading-relaxed"
                  disabled={isStreaming || uploading}
                  rows={1}
                />
              </div>

              {/* Send */}
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isStreaming || uploading}
                size="icon"
                className="h-10 w-10 flex-shrink-0 rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 shadow-md"
                title="Send message"
              >
                {isStreaming ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>

            <p className="text-[11px] text-muted-foreground text-center mt-2">
              📎 Upload docs · Ask questions · Get cited answers from your knowledge base
            </p>
          </div>
        </div>
      </div>

      {/* ─── Document Panel (overlay) ─────────────────────────────────────── */}
      <DocumentPanel
        open={showDocs}
        onClose={() => setShowDocs(false)}
        documents={docsData?.documents ?? []}
        loading={docsLoading}
      />

      {/* ─── Retrieval Pipeline Panel (floating) ──────────────────────────── */}
      <RetrievalPipelinePanel
        trace={retrievalTrace}
        visible={showPipeline}
        onClose={() => setShowPipeline(false)}
      />
    </div>
  );
}
