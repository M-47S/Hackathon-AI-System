import { useState, useRef, useEffect } from "react";
import type { SyntheticEvent, KeyboardEvent } from "react";
import type { ChatMessage, SessionListItem } from "../types/api";
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
    
    const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
    const [hasSubmittedLead, setHasSubmittedLead] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textAreaRef = useRef<HTMLTextAreaElement>(null);

    // 1. Загрузка чата и сброс предыдущего состояния
    useEffect(() => {
        setMessages([]);
        setInput('');
        setSelectedProduct(null);
        setHasSubmittedLead(false);
        setError(null);
        setIsLoading(false); // Индикатор печати гаснет сразу при переходе

        if (urlSessionId) {
            setSessionId(urlSessionId);
            getHistory(urlSessionId)
                .then((data: any) => {
                    const rawMessages = data.history || data.messages || [];
                    const formatted = rawMessages.map((msg: any, index: number) => ({
                        ...msg,
                        id: msg.id || `hist-${index}-${Date.now()}`,
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
                    onSessionCreated({
                        session_id: data.session_id,
                        conversation_summary: null,
                    });
                }
                navigate(`/chat/${data.session_id}`, { replace: true });
            }).catch(err => {
                console.error(err);
                setError("Не удалось создать новый чат");
            });
        }

        setTimeout(() => textAreaRef.current?.focus(), 100);
    }, [urlSessionId]);

    // 2. Скролл вниз при новых сообщениях
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    // 3. Авто-увеличение высоты поля ввода
    useEffect(() => {
        if (textAreaRef.current) {
            textAreaRef.current.style.height = 'auto';
            textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
        }
    }, [input]);

    // 4. Отправка сообщения
    async function handleSend() {
        if (!input.trim() || !sessionId || isLoading) return;

        const targetSessionId = sessionId;
        const userText = input;

        setInput('');
        setError(null);
        setIsLoading(true);
        setSelectedProduct(null); 
        setHasSubmittedLead(false);

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
            
            // Если пользователь переключил чат, пока мы ждали ответ — игнорируем его
            if (window.location.pathname !== `/chat/${targetSessionId}`) {
                return; 
            }

            const assistantMessage: ChatMessage = {
                id: response.message?.id || Date.now().toString(),
                role: 'assistant',
                content: response.message?.content || "Пустой ответ",
                created_at: response.message?.created_at || new Date().toISOString(),
                action: response.action,
                products: response.products || [], 
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
            // Сбрасываем загрузку, только если мы все еще в том же чате
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