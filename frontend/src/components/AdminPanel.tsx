import { useState, useEffect } from 'react';
import { getLeadsList, getLeadDetails } from '../api/admin';
import type { LeadListItem, LeadDetails } from '../api/admin';
import './AdminPanel.css';

export function AdminPanel() {
    const [leads, setLeads] = useState<LeadListItem[]>([]);
    const [selectedLead, setSelectedLead] = useState<LeadDetails | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        getLeadsList()
            .then((data: any) => setLeads(data.items || data.Items || []))
            .catch(err => console.error("Ошибка загрузки заявок:", err))
            .finally(() => setIsLoading(false));
    }, []);

    const handleSelectLead = (id: string) => {
        getLeadDetails(id)
            .then(setSelectedLead)
            .catch(err => console.error(err));
    };

    if (isLoading) return <div className="admin-panel" style={{padding: '40px', color: 'white'}}>Загрузка заявок...</div>;

    return (
        <div className="admin-panel">
            <div className="leads-sidebar">
                <h2>Заявки</h2>
                {leads.length === 0 && <p style={{color: '#aaa'}}>Заявок пока нет</p>}
                {leads.map(lead => (
                    <div 
                        key={lead.lead_id} 
                        className={`lead-item ${selectedLead?.lead_id === lead.lead_id ? 'active' : ''}`}
                        onClick={() => handleSelectLead(lead.lead_id)}
                    >
                        <strong>{lead.company || 'Без компании'}</strong>
                        <span>{lead.problem || 'Нет описания проблемы'}</span>
                        <small>{new Date(lead.created_at).toLocaleString()}</small>
                    </div>
                ))}
            </div>

            <div className="lead-details">
                {selectedLead ? (
                    <>
                        <div style={{ borderBottom: '1px solid #555', paddingBottom: '15px', marginBottom: '20px' }}>
                            <h2 style={{ margin: 0, color: '#e81a0c', fontSize: '1.8rem' }}>
                                {selectedLead.contact.name}
                            </h2>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                            {/* Контакты */}
                            <div style={{ backgroundColor: '#2F3136', padding: '20px', borderRadius: '10px' }}>
                                <h3 style={{ marginTop: 0, fontSize: '1.1rem', borderBottom: '1px solid #444', paddingBottom: '10px' }}>Контакты</h3>
                                <p style={{ margin: '10px 0' }}><strong>Компания:</strong> {selectedLead.contact.company || '—'}</p>
                                <p style={{ margin: '10px 0' }}><strong>Телефон:</strong> {selectedLead.contact.phone || '—'}</p>
                                <p style={{ margin: '10px 0' }}><strong>Email:</strong> {selectedLead.contact.email || '—'}</p>
                            </div>

                            {/* Продукты */}
                            <div style={{ backgroundColor: '#2F3136', padding: '20px', borderRadius: '10px' }}>
                                <h3 style={{ marginTop: 0, fontSize: '1.1rem', borderBottom: '1px solid #444', paddingBottom: '10px' }}>Продукты</h3>
                                <ul style={{ color: '#ddd', paddingLeft: '20px', margin: '10px 0' }}>
                                    {selectedLead.recommended_product_ids?.map(id => (
                                        <li key={id} style={{ marginBottom: '5px' }}>{id}</li>
                                    )) || <li>Нет рекомендаций</li>}
                                </ul>
                            </div>
                        </div>

                        {/* Проблема клиента (теперь под контактами и продуктами) */}
                        <div style={{ marginTop: '20px', backgroundColor: '#2F3136', padding: '20px', borderRadius: '10px', borderLeft: '4px solid #e81a0c' }}>
                            <h3 style={{ marginTop: 0, color: '#aaa', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Проблема клиента</h3>
                            <p style={{ fontSize: '1.1rem', color: 'white', margin: 0, lineHeight: '1.4' }}>
                                {selectedLead.customer_profile?.problem || selectedLead.problem || 'Не указана'}
                            </p>
                        </div>
                        
                        <h3 style={{ marginTop: '30px', borderBottom: '1px solid #555', paddingBottom: '10px', fontSize: '1.2rem' }}>Резюме консультации</h3>
                        <p style={{ color: '#ddd', lineHeight: '1.6', backgroundColor: '#2F3136', padding: '20px', borderRadius: '8px', fontSize: '1rem' }}>
                            {selectedLead.conversation_summary || 'Резюме отсутствует'}
                        </p>
                    </>
                ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#aaa' }}>
                        <p>Выберите заявку из списка слева</p>
                    </div>
                )}
            </div>
        </div>
    );
}