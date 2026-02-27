'use client';

import { ChatInterface } from '@/components/chat-interface';

export default function HomePage() {
  return (
    <div className="h-screen flex flex-col">
      <ChatInterface />
    </div>
  );
}
