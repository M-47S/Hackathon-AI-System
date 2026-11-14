import type { SessionListItem } from './sessions';
import type { ProductCard } from '../types/api';

// Имитация задержки сети
const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

// Заглушка для списка чатов в сайдбаре
export async function mockGetSessionsList(): Promise<{ items: SessionListItem[] }> {
    await delay(500);
    return {
        items: [
            { session_id: 'mock-1', title: 'Проблема с электричеством', created_at: new Date().toISOString() },
            { session_id: 'mock-2', title: 'Учет в магазинах', created_at: new Date().toISOString() },
        ]
    };
}

// Каталог продуктов. Когда бэкенд начнет отдавать полные карточки, 
// мы просто перестанем использовать этот объект.
export const mockProducts: Record<string, ProductCard> = {
    cloud_askue: {
        id: 'cloud_askue',
        name: 'Облачная АСКУЭ',
        description: 'Автоматизированный сбор и обработка показаний приборов учёта.',
    },
    smart_meter: {
        id: 'smart_meter',
        name: 'Умные счетчики',
        description: 'Современные приборы учета с передачей данных по Wi-Fi/LoRa.',
    }
};

// Заглушка для заявок (понадобится для Админки)
export interface MockLead {
    id: string;
    name: string;
    phone: string;
    problem: string;
    status: string;
}

export async function mockGetLeads(): Promise<{ items: MockLead[] }> {
    await delay(500);
    return {
        items: [
            { id: '1', name: 'Иван Иванов', phone: '+7 999 111-22-33', problem: 'Высокие расходы', status: 'new' },
            { id: '2', name: 'ООО Ромашка', phone: '+7 999 444-55-66', problem: 'Нужна автоматизация', status: 'in_progress' },
        ]
    };
}