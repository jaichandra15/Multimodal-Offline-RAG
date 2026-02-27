'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Paperclip, FileText, X, Sparkles, MessageSquare, Trash2, Plus, BookOpen, ChevronDown, ChevronUp, User, Edit2, Check, BarChart3 } from 'lucide-react';
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
import type { Citation } from '@/lib/types';

export function ChatInterface() {
  const [input, setInput] = useState('');
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
  const [currentCitations, setCurrentCitations] = useState<Citation[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadingFileName, setUploadingFileName] = useState('');
  const [showDocuments, setShowDocuments] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  
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

  const { data: documentsData, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(10),
    enabled: showDocuments,
  });
  
  const { data: documentCount } = useQuery({
    queryKey: ['documents-count'],
    queryFn: () => api.getDocuments(1),
  });

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamingMessage]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadingFileName(file.name);

    // Add a system message showing file is uploading
    const uploadStartMessage = {
      role: 'assistant' as const,
      content: `📎 Processing **${file.name}**...`,
    };
    addMessage(uploadStartMessage);

    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await api.uploadFile(file);
      setUploadProgress(100);
      
      // Add success message
      const successMessage = {
        role: 'assistant' as const,
        content: `✅ Successfully uploaded and processed **${file.name}**!\n\n📊 Created ${response.chunks_created} searchable chunks.\n\nYou can now ask questions about this document.`,
      };
      addMessage(successMessage);
      
      toast.success('Document processed successfully!');
      refetch();
    } catch (error: any) {
      // Add error message
      const errorMessage = {
        role: 'assistant' as const,
        content: `❌ Failed to upload **${file.name}**: ${error.message}`,
      };
      addMessage(errorMessage);
      toast.error(`Upload failed: ${error.message}`);
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
      setUploadProgress(0);
      setUploadingFileName('');
      e.target.value = '';
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage = { role: 'user' as const, content: input.trim() };
    addMessage(userMessage);
    setInput('');

    if (useStreaming) {
      setIsStreaming(true);
      setCurrentStreamingMessage('');
      setCurrentCitations([]);

      try {
        await api.chatStream(
          {
            message: userMessage.content,
            conversation_history: messages,
          },
          (chunk) => {
            setCurrentStreamingMessage((prev) => prev + chunk);
          },
          (fullResponse, citations) => {
            addMessage({ role: 'assistant', content: fullResponse, citations });
            setCurrentStreamingMessage('');
            setCurrentCitations([]);
            setIsStreaming(false);
          },
          (error) => {
            toast.error(`Error: ${error.message}`);
            setCurrentStreamingMessage('');
            setCurrentCitations([]);
            setIsStreaming(false);
          },
          (citations) => {
            setCurrentCitations(citations);
          }
        );
      } catch (error: any) {
        toast.error(`Failed to send message: ${error.message}`);
        setIsStreaming(false);
      }
    } else {
      setIsStreaming(true);
      try {
        const response = await api.chat({
          message: userMessage.content,
          conversation_history: messages,
        });
        addMessage({ role: 'assistant', content: response.response, citations: response.citations });
      } catch (error: any) {
        toast.error(`Failed to send message: ${error.message}`);
      } finally {
        setIsStreaming(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - Chat History */}
      {showSidebar && (
        <div className="w-64 border-r bg-muted/30 flex flex-col h-full">
          <div className="p-3 border-b flex-shrink-0">
            <Button
              onClick={() => createConversation()}
              className="w-full justify-start"
              variant="outline"
            >
              <Plus className="h-4 w-4 mr-2" />
              New Chat
            </Button>
          </div>
          <ScrollArea className="flex-1 overflow-y-auto">
            <div className="p-2 space-y-1">
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer hover:bg-muted transition-colors ${
                    currentConversationId === conv.id ? 'bg-muted' : ''
                  }`}
                  onClick={() => {
                    if (editingConversationId !== conv.id) {
                      setCurrentConversation(conv.id);
                    }
                  }}
                >
                  <MessageSquare className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    {editingConversationId === conv.id ? (
                      <input
                        type="text"
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            renameConversation(conv.id, editingTitle);
                            setEditingConversationId(null);
                          } else if (e.key === 'Escape') {
                            setEditingConversationId(null);
                          }
                        }}
                        className="text-sm bg-background border border-border rounded px-2 py-1 w-full"
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <>
                        <p className="text-sm truncate">{conv.title}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatDistanceToNow(conv.updatedAt, { addSuffix: true })}
                        </p>
                      </>
                    )}
                  </div>
                  {editingConversationId === conv.id ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={(e) => {
                        e.stopPropagation();
                        renameConversation(conv.id, editingTitle);
                        setEditingConversationId(null);
                      }}
                    >
                      <Check className="h-3 w-3" />
                    </Button>
                  ) : (
                    <>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 opacity-0 group-hover:opacity-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingConversationId(conv.id);
                          setEditingTitle(conv.title);
                        }}
                      >
                        <Edit2 className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 opacity-0 group-hover:opacity-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteConversation(conv.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </>
                  )}
                </div>
              ))}
              {conversations.length === 0 && (
                <div className="text-center p-8 text-sm text-muted-foreground">
                  No conversations yet
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex-shrink-0">
          <div className="container max-w-4xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-semibold">RAG Assistant</h1>
                  <p className="text-xs text-muted-foreground">
                    Powered by Ollama
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Link href="/metrics">
                  <Button
                    variant="outline"
                    size="sm"
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Metrics
                  </Button>
                </Link>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowDocuments(!showDocuments)}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  {documentCount?.total || 0} Docs
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => createConversation()}
                  disabled={messages.length === 0}
                >
                  New Chat
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="container max-w-4xl mx-auto px-4 py-6">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-4 rounded-2xl mb-4">
                  <Sparkles className="h-12 w-12 text-white" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">
                  How can I help you today?
                </h2>
                <p className="text-muted-foreground max-w-md">
                  Upload documents and ask questions. I'll help you find answers
                  from your knowledge base.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message, index) => (
                  <div key={index}>
                    <div
                      className={`flex gap-4 ${
                        message.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      {message.role === 'assistant' && (
                        <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg h-8 w-8 flex-shrink-0">
                          <Sparkles className="h-4 w-4 text-white" />
                        </div>
                      )}
                      <div className={message.role === 'user' ? 'max-w-[85%]' : 'flex-1 max-w-[85%]'}>
                        <div
                          className={`rounded-xl px-4 py-3 border ${
                            message.role === 'user'
                              ? 'bg-primary text-primary-foreground border-primary'
                              : 'bg-card border-border'
                          }`}
                        >
                          <div className="prose prose-base dark:prose-invert max-w-none">
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
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        </div>
                        
                        {/* Citations Display */}
                        {message.role === 'assistant' && message.citations && message.citations.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground px-2">
                              <BookOpen className="h-3 w-3" />
                              <span>{message.citations.length} source{message.citations.length > 1 ? 's' : ''}</span>
                            </div>
                            {message.citations.map((citation) => {
                              // Extract filename from source
                              const filename = citation.document_source.split('/').pop() || citation.document_source;
                              // Extract page number from metadata if available
                              const pageNum = citation.metadata?.page || citation.metadata?.page_number;
                              
                              return (
                                <Card key={citation.number} className="p-2">
                                  <div className="flex items-center gap-2">
                                    <div className="flex-shrink-0 h-5 w-5 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">
                                      {citation.number}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-1 text-sm">
                                        <span className="font-medium truncate">{filename}</span>
                                        {pageNum && (
                                          <span className="text-muted-foreground flex-shrink-0">• Page {pageNum}</span>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                </Card>
                              );
                            })}
                          </div>
                        )}
                      </div>
                      {message.role === 'user' && (
                        <div className="bg-primary p-2 rounded-lg h-8 w-8 flex-shrink-0 flex items-center justify-center">
                          <User className="h-4 w-4 text-primary-foreground" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {currentStreamingMessage && (
                  <div>
                    <div className="flex gap-4 justify-start">
                      <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg h-8 w-8 flex-shrink-0">
                        <Sparkles className="h-4 w-4 text-white" />
                      </div>
                      <div className="flex-1 max-w-[85%]">
                        <div className="rounded-xl px-4 py-3 bg-card border border-border">
                          <div className="prose prose-base dark:prose-invert max-w-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {currentStreamingMessage}
                            </ReactMarkdown>
                          </div>
                        </div>
                        
                        {/* Current Citations During Streaming */}
                        {currentCitations.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground px-2">
                              <BookOpen className="h-3 w-3" />
                              <span>{currentCitations.length} source{currentCitations.length > 1 ? 's' : ''}</span>
                            </div>
                            {currentCitations.map((citation) => {
                              const filename = citation.document_source.split('/').pop() || citation.document_source;
                              const pageNum = citation.metadata?.page || citation.metadata?.page_number;
                              
                              return (
                                <Card key={citation.number} className="p-2">
                                  <div className="flex items-center gap-2">
                                    <div className="flex-shrink-0 h-5 w-5 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">
                                      {citation.number}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-1 text-sm">
                                        <span className="font-medium truncate">{filename}</span>
                                        {pageNum && (
                                          <span className="text-muted-foreground flex-shrink-0">• Page {pageNum}</span>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                </Card>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div ref={scrollRef} />
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex-shrink-0">
          <div className="container max-w-4xl mx-auto px-4 py-4">
            {uploading && (
              <div className="mb-3 p-4 bg-muted rounded-xl border border-border">
                <div className="flex items-center gap-3 mb-3">
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {uploadingFileName}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Processing document...
                    </p>
                  </div>
                  <span className="text-sm font-medium">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-1.5" />
              </div>
            )}
            <div className="relative">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message RAG Assistant..."
                className="min-h-[60px] pr-24 resize-none rounded-2xl"
                disabled={isStreaming || uploading}
              />
              <div className="absolute right-2 bottom-2 flex gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileUpload}
                  disabled={uploading}
                  accept=".pdf,.docx,.pptx,.xlsx,.md,.txt,.mp3,.wav,.m4a,.flac"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-lg"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || isStreaming || uploading}
                  size="icon"
                  className="h-8 w-8 rounded-lg"
                >
                  {isStreaming ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-2">
              Upload documents with the paperclip or ask questions directly
            </p>
          </div>
        </div>
      </div>

      {/* Sidebar - Documents */}
      {showDocuments && (
        <div className="w-80 border-l bg-muted/30 flex flex-col h-full">
          <div className="p-4 border-b flex items-center justify-between flex-shrink-0">
            <h3 className="font-semibold">Documents</h3>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowDocuments(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-2">
              {documentsData?.documents.map((doc) => (
                <Card key={doc.id} className="p-3">
                  <div className="flex items-start gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{doc.title}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        {doc.source}
                      </p>
                    </div>
                  </div>
                </Card>
              ))}
              {documentsData?.documents.length === 0 && (
                <div className="text-center p-8 text-sm text-muted-foreground">
                  No documents yet. Upload one to get started!
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );
}
