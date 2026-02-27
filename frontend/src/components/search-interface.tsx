'use client';

import { useState } from 'react';
import { Search as SearchIcon, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import type { SearchResultItem } from '@/lib/types';

export function SearchInterface() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      const response = await api.search({ query, limit: 10 });
      setResults(response.results);
      if (response.results.length === 0) {
        toast.info('No results found');
      }
    } catch (error: any) {
      toast.error(`Search failed: ${error.message}`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search knowledge base..."
          disabled={isSearching}
        />
        <Button onClick={handleSearch} disabled={isSearching || !query.trim()}>
          {isSearching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <SearchIcon className="h-4 w-4" />
          )}
        </Button>
      </div>

      <ScrollArea className="h-[600px]">
        <div className="space-y-4">
          {results.map((result, index) => (
            <Card key={result.chunk_id}>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-start">
                    <h4 className="font-medium text-sm">
                      {result.document_title}
                    </h4>
                    <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                      {(result.similarity * 100).toFixed(1)}% match
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {result.document_source}
                  </p>
                  <p className="text-sm mt-2 line-clamp-4">{result.content}</p>
                </div>
              </CardContent>
            </Card>
          ))}

          {results.length === 0 && !isSearching && query && (
            <div className="text-center p-8 text-muted-foreground">
              No results found
            </div>
          )}

          {results.length === 0 && !isSearching && !query && (
            <div className="text-center p-8 text-muted-foreground">
              Enter a search query to find relevant documents
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
