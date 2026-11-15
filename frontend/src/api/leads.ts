import { http } from './client';

export interface LeadCreateRequest {
    session_id: string;
    submission_id: string;
    contact: {
        name: string;
        company?: string | null;
        phone?: string | null;
        email?: string | null;
    };
    consent: boolean;
}

// Возвращаем Promise<void>, так как ответ нам не нужен
export function createLead(data: LeadCreateRequest): Promise<void> {
    return http.post<void>('/api/leads', data);
}