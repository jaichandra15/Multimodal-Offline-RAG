'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Upload, Loader2, FileText, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { formatDistanceToNow } from 'date-fns';

export function DocumentManager() {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const { data: documentsData, isLoading, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(),
  });

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await api.uploadFile(file);
      setUploadProgress(100);
      toast.success(response.message);
      refetch();
    } catch (error: any) {
      toast.error(`Upload failed: ${error.message}`);
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
      setUploadProgress(0);
      e.target.value = '';
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Upload Document</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileUpload}
                disabled={uploading}
                accept=".pdf,.docx,.pptx,.xlsx,.md,.txt,.mp3,.wav,.m4a,.flac"
              />
              <label htmlFor="file-upload" className="flex-1">
                <Button
                  type="button"
                  className="w-full"
                  disabled={uploading}
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Choose File
                    </>
                  )}
                </Button>
              </label>
            </div>

            {uploading && (
              <div className="space-y-2">
                <Progress value={uploadProgress} />
                <p className="text-sm text-muted-foreground text-center">
                  {uploadProgress}%
                </p>
              </div>
            )}

            <p className="text-sm text-muted-foreground">
              Supported: PDF, Word, PowerPoint, Excel, Markdown, Text, Audio (MP3, WAV)
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            Documents ({documentsData?.total || 0})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center p-8">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <div className="space-y-2">
                {documentsData?.documents.map((doc) => (
                  <Card key={doc.id} className="p-4">
                    <div className="flex items-start gap-3">
                      <FileText className="h-5 w-5 text-muted-foreground mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium truncate">{doc.title}</h4>
                        <p className="text-sm text-muted-foreground truncate">
                          {doc.source}
                        </p>
                        <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                          <Calendar className="h-3 w-3" />
                          {formatDistanceToNow(new Date(doc.created_at), {
                            addSuffix: true,
                          })}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}

                {documentsData?.documents.length === 0 && (
                  <div className="text-center p-8 text-muted-foreground">
                    No documents uploaded yet
                  </div>
                )}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
