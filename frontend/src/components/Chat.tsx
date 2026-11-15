import { useState, useRef, useEffect } from "react";
import type { SyntheticEvent, KeyboardEvent } from "react";
import type { ChatMessage, SessionListItem, ProductCard as ProductType } from "../types/api";
import { createSession, getHistory } from "../api/sessions";
import { sendMessage as apiSendMessage } from "../api/chat";
import { useNavigate, useParams } from "react-router-dom";
import { ProductCard } from "./ProductCard";
import { LeadForm } from "./LeadForm";
import './Chat.css';

interface ChatProps {
    isDebugMode: boolean;
    onSessionCreated?: (session: SessionListItem) => void;
    onSessionUpdated?: (sessionId: string, summary: string) => void;
}

export function Chat({ isDebugMode, onSessionCreated, onSessionUpdated }: ChatProps) {
    const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
    const navigate = useNavigate();
    
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const [isLeadFormOpen, setIsLeadFormOpen] = useState(false);
    const [selectedProduct, setSelectedProduct] = useState<ProductType | null>(null);
    const [hasSubmittedLead, setHasSubmittedLead] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        setMessages([]);
        setInput('');
        setHasSubmittedLead(false);
        setError(null);
        setIsLoading(false); 

        const savedProductStr = localStorage.getItem(`lead_product_${urlSessionId}`);
        if (savedProductStr) {
            try {
                const prod = JSON.parse(savedProductStr);
                setSelectedProduct(prod);
                setIsLeadFormOpen(true);
            } catch (e) {
                localStorage.removeItem(`lead_product_${urlSessionId}`);
                setIsLeadFormOpen(false);
            }
        } else {
            setIsLeadFormOpen(false);
        }

        if (urlSessionId) {
            setSessionId(urlSessionId);
            getHistory(urlSessionId)
                .then((data: any) => {
                    const rawMessages = data.history || data.messages || [];
                    const formatted = rawMessages.map((msg: any, index: number) => ({
                        ...msg,
                        id: msg.id || `hist-${index}-${Date.now()}`,
                        // Защита: если бэк прислал продукты в recommended_product_ids, берем их
                        recommended_product_ids: msg.recommended_product_ids || [], 
                    }));
                    setMessages(formatted);
                })
                .catch(err => {
                    console.error(err);
                    setError("Не удалось загрузить историю чата");
                });
        } else {
            createSession().then(data => {
                setSessionId(data.session_id);
                if (onSessionCreated) {
                    onSessionCreated({ session_id: data.session_id, conversation_summary: null });
                }
                navigate(`/chat/${data.session_id}`, { replace: true });
            }).catch(err => {
                console.error(err);
                setError("Не удалось создать новый чат");
            });
        }

        setTimeout(() => textAreaRef.current?.focus(), 100);
    }, [urlSessionId]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    useEffect(() => {
        if (textAreaRef.current) {
            textAreaRef.current.style.height = 'auto';
            textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
        }
    }, [input]);

    async function handleSend() {
        if (!input.trim() || !sessionId || isLoading) return;

        const targetSessionId = sessionId;
        const userText = input;

        setInput('');
        setError(null);
        setIsLoading(true);
        setIsLeadFormOpen(false); 
        setSelectedProduct(null);
        setHasSubmittedLead(false);
        localStorage.removeItem(`lead_product_${targetSessionId}`);

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: userText,
            created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);

        try {
            const response = await apiSendMessage({
                session_id: targetSessionId,
                request_id: Date.now().toString(),
                message: userText,
            });
            
            if (window.location.pathname !== `/chat/${targetSessionId}`) return;

            // ИСПРАВЛЕНИЕ: Достаем продукты из ответа (на случай, если бэк прислал их на верхнем уровне)
            const products = response.recommended_product_ids || (response as any).products || [];

            const assistantMessage: ChatMessage = {
                id: response.message?.id || Date.now().toString(),
                role: 'assistant',
                content: response.message?.content || "Пустой ответ",
                created_at: response.message?.created_at || new Date().toISOString(),
                action: response.action,
                // Записываем продукты в объект сообщения, чтобы React их увидел
                recommended_product_ids: products.length > 0 ? products : null,
                sequence: response.sequence || null,
                used_documents: response.used_documents || null,
                debugRaw: JSON.stringify(response, null, 2),
            };

            setMessages((prev) => [...prev, assistantMessage]);

            if (response.conversation_summary && onSessionUpdated) {
                onSessionUpdated(targetSessionId, response.conversation_summary);
            }
        } catch (err: any) {
            if (window.location.pathname === `/chat/${targetSessionId}`) {
                setMessages((prev) => [...prev, { 
                    id: Date.now().toString(), 
                    role: 'assistant', 
                    content: 'Ошибка: ' + (err.message || 'Неизвестная ошибка'), 
                    created_at: new Date().toISOString() 
                }]);
            }
        } finally {
            if (window.location.pathname === `/chat/${targetSessionId}`) {
                setIsLoading(false);
                setTimeout(() => textAreaRef.current?.focus(), 0);
            }
        }
    }
    
    function handleFormSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        handleSend();
    }

    function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    }

    const handleLeadSubmitSuccess = () => {
        setIsLeadFormOpen(false);
        setSelectedProduct(null);
        setHasSubmittedLead(true);
        localStorage.removeItem(`lead_product_${sessionId}`);
        setMessages(prev => [...prev, {
            id: Date.now().toString(),
            role: 'assistant',
            content: 'Спасибо! Заявка принята, менеджер свяжется с вами.',
            created_at: new Date().toISOString()
        }]);
    }

    return (
        <div className="chat-container">
            <main className="chat-messages">
                 {messages.map((msg) => (
                    <div key={msg.id} className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                        {msg.content}

                        {/* Отрисовка продуктов */}
                        {msg.recommended_product_ids && msg.recommended_product_ids.length > 0 && (
                            <div className="products-container">
                                {msg.recommended_product_ids.map((product) => (
                                    <ProductCard 
                                        key={product.id} 
                                        product={product} 
                                        onInterested={() => {
                                            setSelectedProduct(product);
                                            setIsLeadFormOpen(true);
                                            localStorage.setItem(`lead_product_${sessionId}`, JSON.stringify(product));
                                        }}
                                    />
                                ))}
                            </div>
                        )}

                        {/* Блок отладки */}
                        {isDebugMode && msg.role === 'assistant' && (
                            <div className="debug-panel" style={{ marginTop: '15px', borderTop: '1px dashed #555', paddingTop: '10px' }}>
                                {msg.sequence ? (
                                    <details className="debug-json" open>
                                        <summary style={{ color: '#e81a0c', fontWeight: 'bold', cursor: 'pointer' }}>🧠 Цепочка размышлений (Sequence)</summary>
                                        <pre style={{ whiteSpace: 'pre-wrap', color: '#00ffcc', marginTop: '10px', fontSize: '0.85rem' }}>
                                            {msg.sequence}
                                        </pre>
                                    </details>
                                ) : null}
                                
                                {msg.used_documents && msg.used_documents.length > 0 ? (
                                    <details className="debug-json" open style={{ marginTop: '10px' }}>
                                        <summary style={{ color: '#007aff', fontWeight: 'bold', cursor: 'pointer' }}>📚 Использованные документы</summary>
                                        <ul style={{ color: '#fff', marginTop: '10px', paddingLeft: '20px', fontSize: '0.85rem' }}>
                                            {msg.used_documents.map((doc, i) => (
                                                <li key={i} style={{ marginBottom: '5px' }}>{doc}</li>
                                            ))}
                                        </ul>
                                    </details>
                                ) : null}

                                {!msg.sequence && (!msg.used_documents || msg.used_documents.length === 0) && msg.debugRaw ? (
                                    <details className="debug-json">
                                        <summary style={{ color: '#aaa', cursor: 'pointer' }}>Сырой JSON ответа</summary>
                                        <pre style={{ whiteSpace: 'pre-wrap', color: '#00ffcc', marginTop: '10px', fontSize: '0.85rem' }}>
                                            {msg.debugRaw}
                                        </pre>
                                    </details>
                                ) : null}
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && <div className="message-bubble message-assistant">Печатает...</div>}
                <div ref={messagesEndRef} />
            </main>
            
            {isLeadFormOpen && !hasSubmittedLead && (
                <div className="lead-form-container">
                    {selectedProduct && (
                        <div style={{ color: 'white', marginBottom: '10px', fontSize: '0.9rem' }}>
                            Заявка на продукт: <strong>{selectedProduct.name}</strong>
                        </div>
                    )}
                    <LeadForm
                        sessionId={sessionId || ''}
                        onSubmitSuccess={handleLeadSubmitSuccess}
                        onCancel={() => {
                            setIsLeadFormOpen(false);
                            setSelectedProduct(null);
                            localStorage.removeItem(`lead_product_${sessionId}`);
                        }}
                    />
                </div>
            )}
            {error && <p style={{color: '#ff4d4f', textAlign: "center", padding: '0 20px'}}>{error}</p>}

            <form onSubmit={handleFormSubmit} className="chat-input-form">
                <textarea 
                    ref={textAreaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Введите текст"
                    className="chat-input"
                    rows={1}
                    onKeyDown={handleKeyDown}
                />
                <button type="submit" className="chat-submit" disabled={isLoading || !input.trim() || !sessionId}>
                    Отправить
                </button>
            </form>
        </div>
    );
}