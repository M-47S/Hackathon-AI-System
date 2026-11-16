import { http } from './client';

// Тип для элемента списка заявок (то, что приходит в GET /api/admin/leads)
export interface LeadListItem {
    lead_id: string;
    company: string | null;
    status: string;
    problem: string | null;
    recommended_product_ids: string[];
    created_at: string;
}

// Тип для подробной информации о заявке (то, что приходит в GET /api/admin/leads/{id})
export interface LeadDetails extends LeadListItem {
    session_id: string;
    contact: {
        name: string;
        company: string | null;
        phone: string | null;
        email: string | null;
    };
    customer_profile: any; // Пока оставляем any, если бэк не типизировал профиль
    conversation_summary: string | null;
}

// Функция для получения списка всех заявок (доступна только админу)
export function getLeadsList(): Promise<{ items: LeadListItem[] }> {
    // Бэкендер может назвать поле "items" или "Items" (с большой буквы),
    // но TypeScript мы обманываем через Promise, а в компоненте AdminPanel уже обрабатываем оба варианта
    return http.get<{ items: LeadListItem[] }>('/api/admin/leads');
}

// Функция для получения деталей конкретной заявки
export function getLeadDetails(leadId: string): Promise<LeadDetails> {
    return http.get<LeadDetails>(`/api/admin/leads/${leadId}`);
}