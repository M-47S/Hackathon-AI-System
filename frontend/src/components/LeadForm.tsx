import { useState, useEffect } from 'react';
import type { SyntheticEvent } from 'react';
import { createLead } from '../api/leads';
import './LeadForm.css';

interface LeadFormProps {
    sessionId: string;
    onSubmitSuccess: () => void;
    onCancel: () => void;
}

export function LeadForm({ sessionId, onSubmitSuccess, onCancel }: LeadFormProps) {
    // Восстанавливаем данные из localStorage при монтировании
    const savedData = JSON.parse(localStorage.getItem(`lead_data_${sessionId}`) || '{}');
    
    const [name, setName] = useState(savedData.name || '');
    const [company, setCompany] = useState(savedData.company || '');
    const [phone, setPhone] = useState(savedData.phone || '');
    const [email, setEmail] = useState(savedData.email || '');
    
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    // Сохраняем данные в localStorage при каждом изменении
    useEffect(() => {
        localStorage.setItem(`lead_data_${sessionId}`, JSON.stringify({ name, company, phone, email }));
    }, [name, company, phone, email, sessionId]);

    async function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        
        const phoneRegex = /^\+?[0-9\s\-()]{10,20}$/;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (!phone.trim() && !email.trim()) {
            setError('Укажите телефон или email');
            return;
        }
        if (phone.trim() && !phoneRegex.test(phone)) {
            setError('Неверный формат телефона');
            return;
        }
        if (email.trim() && !emailRegex.test(email)) {
            setError('Неверный формат email');
            return;
        }

        setError(null);
        setIsLoading(true);

        try {
            await createLead({
                session_id: sessionId,
                submission_id: Date.now().toString(),
                contact: { name, company: company || null, phone: phone || null, email: email || null },
                consent: true, 
            });

            localStorage.removeItem(`lead_data_${sessionId}`);
            onSubmitSuccess();
        } catch (err: any) {
            setError(err.message || 'Ошибка при отправке заявки');
        } finally {
            setIsLoading(false);
        }
    }

    function handleCancel() {
        localStorage.removeItem(`lead_data_${sessionId}`);
        onCancel();
    }

    return (
        <form className="lead-form" onSubmit={handleSubmit}>
            <h4>Оставить заявку</h4>
            <p>Менеджер свяжется с вами для уточнения деталей.</p>
            
            {error && <div style={{ color: '#ff4d4f', fontSize: '0.9rem', marginBottom: '10px' }}>{error}</div>}

            <input type="text" placeholder="Ваше имя *" value={name} onChange={(e) => setName(e.target.value)} required />
            <input type="text" placeholder="Компания" value={company} onChange={(e) => setCompany(e.target.value)} />
            <input type="tel" placeholder="Телефон" value={phone} onChange={(e) => setPhone(e.target.value)} />
            <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
            
            <div className="lead-actions">
                <button type="button" onClick={handleCancel} className="cancel-btn" disabled={isLoading}>
                    Отмена
                </button>
                <button type="submit" className="submit-btn" disabled={isLoading || !name.trim()}>
                    {isLoading ? 'Отправка...' : 'Отправить'}
                </button>
            </div>
        </form>
    );
}