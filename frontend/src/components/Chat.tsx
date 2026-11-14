import { useState, useRef, useEffect } from "react";
import type { SyntheticEvent, KeyboardEvent } from "react";
import type { ChatMessage } from "../types/api";
import { createSession, getHistory } from "../api/sessions";
import { sendMessage as apiSendMessage } from "../api/chat";
import { useNavigate, useParams } from "react-router-dom";
import { mockProducts } from "../api/mock";
import { ProductCard } from "./ProductCard";
import { LeadForm } from "./LeadForm";
import './Chat.css';

// НОВОЕ: Пропс для режима отладки
interface ChatProps {
    isDebugMode: boolean;
}

export function Chat({ isDebugMode }: ChatProps) {
    const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
    const navigate = useNavigate();
    
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [hasSubmittedLead, setHasSubmittedLead] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (urlSessionId) {
            setSessionId(urlSessionId);
            getHistory(urlSessionId)
                .then(data => setMessages(data.messages || []))
                .catch(console.error);
        } else {
            createSession().then(data => {
                setSessionId(data.session_id);
                navigate(`/chat/${data.session_id}`, { replace: true });
            }).catch(err => console.error(err));
        }
    }, [urlSessionId]); // Убрали navigate, чтобы не было бесконечного цикла

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

        const userText = input;
        setInput('');
        setError(null);
        setIsLoading(true);
        setSelectedProduct(null); 

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: userText,
            created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);

        try {
            const response = await apiSendMessage({
                session_id: sessionId,
                request_id: Date.now().toString(),
                message: userText,
            });
            
            //ЗАГЛУШКА
            const rawIds = (response as any).product_ids || [];
            const foundProducts = rawIds.map((id: string) => mockProducts[id]).filter(Boolean);
            
            const assistantMessage: ChatMessage = {
                id: Date.now().toString() + '-assistant',
                role: 'assistant',
                content: (response as any).answer,
                created_at: new Date().toISOString(),
                action: response.action,
                products: foundProducts,
                // НОВОЕ: Сохраняем сырой JSON для отладки
                debugRaw: JSON.stringify(response, null, 2), 
            };

            setMessages((prev) => [...prev, assistantMessage]);
        } catch (err: any) {
            setMessages((prev) => [...prev, { id: Date.now().toString(), role: 'assistant', content: 'Ошибка: ' + err.message, created_at: new Date().toISOString() }]);
        } finally {
            setIsLoading(false);
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
        setSelectedProduct(null);
        setHasSubmittedLead(true);
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

                        {msg.products && msg.products.length > 0 && (
                            <div className="products-container">
                                {msg.products.map((product) => (
                                    <ProductCard 
                                        key={product.id} 
                                        product={product} 
                                        onInterested={(name) => setSelectedProduct(name)}
                                    />
                                ))}
                            </div>
                        )}

                        {/* НОВОЕ: Блок отладки */}
                        {isDebugMode && msg.debugRaw && (
                            <details className="debug-json">
                                <summary>Показать JSON ответа</summary>
                                <pre>{msg.debugRaw}</pre>
                            </details>
                        )}
                    </div>
                ))}
                {isLoading && <div className="message-bubble message-assistant">Печатает...</div>}
                <div ref={messagesEndRef} />
            </main>
            
            {selectedProduct && !hasSubmittedLead && (
                <div className="lead-form-container">
                    <LeadForm
                        sessionId={sessionId || ''}
                        onSubmitSuccess={handleLeadSubmitSuccess}
                        onCancel={() => setSelectedProduct(null)}
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
                    disabled={isLoading || !sessionId}
                />
                <button 
                    type="submit"
                    className="chat-submit"
                    disabled={isLoading || !input.trim() || !sessionId}
                >
                    Отправить
                </button>
            </form>
        </div>
    );
}