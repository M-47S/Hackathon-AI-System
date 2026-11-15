export type Action =
  | 'answer_question'
  | 'ask_question'
  | 'recommend'
  | 'offer_handoff'
  | 'out_of_scope';

export type MessageRole = 'user' | 'assistant';

export interface ProductCard {
  id: string;
  name: string;
  description: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  action?: Action | null;
  recommended_product_ids?: ProductCard[] | null;
  debugRaw?: string; 
  // Поля для отладки (сохраняются только в стейте React)
  sequence?: string | null;
  used_documents?: string[] | null;
}

export interface ChatRequest {
  session_id: string;
  request_id: string;
  message: string;
}

export interface ChatResponse {
  session_id: string;
  request_id: string;
  message: ChatMessage;
  action: Action;
  recommended_product_ids?: ProductCard[] | null;
  conversation_summary?: string | null;
  // Поля для отладки
  sequence?: string | null;
  used_documents?: string[] | null;
}

export interface SessionResponse {
  session_id: string;
  status: string;
  created_at: string;
}

export interface HistoryResponse {
  session_id: string;
  messages: ChatMessage[];
}

export interface SessionListItem {
  session_id: string;
  conversation_summary: string | null;
}