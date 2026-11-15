export type Action =
  | 'answer_question'
  | 'ask_question'
  | 'recommend'
  | 'offer_handoff'
  | 'out_of_scope';

export type MessageRole = 'user' | 'assistant';

export interface CustomerProfile {
  business_type: string | null;
  objects_count: number | null;
  problem: string | null;
  metering_method: string | null;
}

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
  products?: ProductCard[];
  debugRaw?: string; 
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
  products: ProductCard[];
  conversation_summary?: string | null; // Добавили, так как бэкенд его присылает
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

// НОВОЕ: Тип для элемента списка сессий
export interface SessionListItem {
  session_id: string;
  conversation_summary: string | null;
}