import { useState } from 'react';
import type { SyntheticEvent } from 'react';
import { createLead } from '../api/leads';
import './LeadForm.css';

interface LeadFormProps {
    sessionId: string;
    onSubmitSuccess: () => void;
    onCancel: () => void;
}

export function LeadForm({ sessionId, onSubmitSuccess, onCancel }: LeadFormProps) {
    const [name, setName] = useState('');
    const [company, setCompany] = useState('');
    const [phone, setPhone] = useState('');
    const [email, setEmail] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    async function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        if (!phone.trim() && !email.trim()) {
            setError('Укажите телефон или email');
            return;
        }

        setError(null);
        setIsLoading(true);

        try {
            await createLead({
                session_id: sessionId,
                submission_id: Date.now().toString(), // Защита от дублей
                contact: {
                    name,
                    company: company || null,
                    phone: phone || null,
                    email: email || null,
                },
                consent: true, 
            });

            // Если ошибки нет (запрос прошел), вызываем успех
            onSubmitSuccess();
        } catch (err: any) {
            setError(err.message || 'Ошибка при отправке заявки');
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <form className="lead-form" onSubmit={handleSubmit}>
            <h4>Оставить заявку</h4>
            <p>Менеджер свяжется с вами для уточнения деталей.</p>
            
            {error && <div style={{ color: 'red', fontSize: '0.9rem' }}>{error}</div>}

            <input 
                type="text" 
                placeholder="Ваше имя *" 
                value={name} 
                onChange={(e) => setName(e.target.value)} 
                required 
            />
            <input 
                type="text" 
                placeholder="Компания" 
                value={company} 
                onChange={(e) => setCompany(e.target.value)} 
            />
            <input 
                type="tel" 
                placeholder="Телефон" 
                value={phone} 
                onChange={(e) => setPhone(e.target.value)} 
            />
            <input 
                type="email" 
                placeholder="Email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
            />
            
            <div className="lead-actions">
                <button type="button" onClick={onCancel} className="cancel-btn" disabled={isLoading}>
                    Отмена
                </button>
                <button type="submit" className="submit-btn" disabled={isLoading || !name.trim()}>
                    {isLoading ? 'Отправка...' : 'Отправить'}
                </button>
            </div>
        </form>
    );
}