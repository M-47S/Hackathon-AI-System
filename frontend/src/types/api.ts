//Типы ответов
export type Action = 
    | 'answer_question'
    | 'ask_question'
    | 'recommend'
    | 'offer_handoff'
    | 'out_of_scope';

export type MessageRole = 'user' | 'assistant';

//Профиль клиента 
export interface CustomerProfile {
    business_type: string | null;
    object_count: number | null;
    problem: string | null;
    metering_method: string | null; 
}

//Карточка продукта
export interface ProductCard {
    id: string;
    name: string;
    description: string;
}

//Сообщение в чате
export interface ChatMessage {
    id: string;
    role: MessageRole;
    content: string;
    created_at: string;
    action?: Action | null;
    products?: ProductCard[];
}

//Запрос на отправку сообщения
export interface ChatResponse {
    session_id: string;
    request_id: string;
    message: ChatMessage;
    action: Action;
    products: ProductCard[];
}

//Ответ при создании сессии
export interface SessionResponse {
    session_id: string;
    status: string;
    created_at: string;
}

export interface HistoryResponse {
    session_id: string;
    messages: ChatMessage[];
}