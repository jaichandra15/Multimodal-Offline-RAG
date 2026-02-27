// Global State Management with Zustand

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage } from './types';

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

interface ChatStore {
  conversations: Conversation[];
  currentConversationId: string | null;
  isStreaming: boolean;
  
  // Conversation management
  createConversation: () => void;
  deleteConversation: (id: string) => void;
  setCurrentConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  
  // Message management
  addMessage: (message: ChatMessage) => void;
  setMessages: (messages: ChatMessage[]) => void;
  clearMessages: () => void;
  
  // Streaming
  setIsStreaming: (isStreaming: boolean) => void;
  
  // Getters
  getCurrentMessages: () => ChatMessage[];
  getCurrentConversation: () => Conversation | undefined;
}

const generateId = () => Math.random().toString(36).substring(7);

const generateTitle = (messages: ChatMessage[]): string => {
  const firstUserMessage = messages.find(m => m.role === 'user');
  if (!firstUserMessage) return 'New Chat';
  return firstUserMessage.content.slice(0, 50) + (firstUserMessage.content.length > 50 ? '...' : '');
};

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      currentConversationId: null,
      isStreaming: false,
      
      createConversation: () => {
        const id = generateId();
        const newConversation: Conversation = {
          id,
          title: 'New Chat',
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        set((state) => ({
          conversations: [newConversation, ...state.conversations],
          currentConversationId: id,
        }));
      },
      
      deleteConversation: (id) => {
        set((state) => {
          const newConversations = state.conversations.filter(c => c.id !== id);
          const newCurrentId = state.currentConversationId === id 
            ? (newConversations[0]?.id || null)
            : state.currentConversationId;
          return {
            conversations: newConversations,
            currentConversationId: newCurrentId,
          };
        });
      },
      
      setCurrentConversation: (id) => {
        set({ currentConversationId: id });
      },
      
      renameConversation: (id, title) => {
        set((state) => ({
          conversations: state.conversations.map(conv =>
            conv.id === id ? { ...conv, title, updatedAt: Date.now() } : conv
          ),
        }));
      },
      
      addMessage: (message) => {
        set((state) => {
          const currentId = state.currentConversationId;
          if (!currentId) {
            // Create new conversation if none exists
            const id = generateId();
            const newConversation: Conversation = {
              id,
              title: message.role === 'user' ? generateTitle([message]) : 'New Chat',
              messages: [message],
              createdAt: Date.now(),
              updatedAt: Date.now(),
            };
            return {
              conversations: [newConversation, ...state.conversations],
              currentConversationId: id,
            };
          }
          
          const updatedConversations = state.conversations.map(conv => {
            if (conv.id === currentId) {
              const newMessages = [...conv.messages, message];
              return {
                ...conv,
                messages: newMessages,
                title: conv.messages.length === 0 && message.role === 'user' 
                  ? generateTitle(newMessages)
                  : conv.title,
                updatedAt: Date.now(),
              };
            }
            return conv;
          });
          
          return { conversations: updatedConversations };
        });
      },
      
      setMessages: (messages) => {
        set((state) => {
          const currentId = state.currentConversationId;
          if (!currentId) return state;
          
          const updatedConversations = state.conversations.map(conv =>
            conv.id === currentId
              ? { ...conv, messages, updatedAt: Date.now() }
              : conv
          );
          
          return { conversations: updatedConversations };
        });
      },
      
      clearMessages: () => {
        get().createConversation();
      },
      
      setIsStreaming: (isStreaming) => set({ isStreaming }),
      
      getCurrentMessages: () => {
        const state = get();
        const current = state.conversations.find(c => c.id === state.currentConversationId);
        return current?.messages || [];
      },
      
      getCurrentConversation: () => {
        const state = get();
        return state.conversations.find(c => c.id === state.currentConversationId);
      },
    }),
    {
      name: 'chat-storage',
    }
  )
);

interface SettingsStore {
  useStreaming: boolean;
  setUseStreaming: (useStreaming: boolean) => void;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      useStreaming: true,
      setUseStreaming: (useStreaming) => set({ useStreaming }),
    }),
    {
      name: 'settings-storage',
    }
  )
);
