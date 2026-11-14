import { useState } from "react";
import type { SyntheticEvent } from "react";
import './LeadForm.css'


interface LeadFormProps {
    sessionId: string;
    onSubmitSuccess: () => void;
    onCancel: () => void;
}

export function LeadForm({ sessionId, onSubmitSuccess, onCancel }: LeadFormProps) {
    const [name, setName] = useState('');
    const [phone, setPhone] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    async function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        setIsLoading(true);
        
        //ЗАГЛУШКА
        console.log('Отправка заявки:', {sessionId, name, phone});
        setTimeout(() => {
            setIsLoading(false);
            onSubmitSuccess();
        }, 1000);


    }

    return (
        <form className="lead-form" onSubmit={handleSubmit}>
            <h4>Оставить заявку</h4>
            <p>Менеджер свяжется с вами для уточнения деталей</p>

            <input 
                type="text"
                placeholder="Ваше имя"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
            />
            <input 
                type="text"
                placeholder="Телефон"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                required
            />

            <div className="lead-action">
                <button type="button" onClick={onCancel} className="cancel-btn">
                    Отмена
                </button>
                <button type="submit" disabled={isLoading} className="submit-btn">
                    {isLoading ? 'Отправка' : 'Отправить'}
                </button>
            </div>
        </form>
    );
}