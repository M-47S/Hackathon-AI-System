import { useState, useRef, useEffect, type SyntheticEvent, type KeyboardEvent } from "react";
import type { ChatMessage } from "../types/api";
import { createSession } from "../api/sessions";
import { sendMessage as apiSendMessage } from "../api/chat";

import './Chat.css'


export function Chat() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        async function initSession() {
            try {
                const data = await createSession();
                setSessionId(data.session_id);
            } catch (err: any) {
                setError(err.message || 'Не удалось создать сессию')
            }
        }
        initSession();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        if (textAreaRef.current){
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

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: userText,
            created_at: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, userMessage])

        try {
            const response = await apiSendMessage({
                session_id: sessionId,
                request_id: Date.now().toString(),
                message: userText,
            });

            const assistantMessage: ChatMessage = {
                id: response.message.id,
                role: 'assistant',
                content: response.message.content,
                created_at: response.message.created_at,
                action: response.action,
                products: response.products,
            };

            setMessages((prev) => [...prev, assistantMessage]);
        } catch (err: any) {
            setError(err.message || 'Ошибка при отправке сообщения');
            setInput(userText);
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


    return (
        <div className="chat-container">

            <main className="chat-messages">
                {messages.length === 0 && !isLoading ? (
                    <p className="chat-empty">
                        {sessionId ? 'Начните диалог' : 'Подключение к серверу'}
                    </p> 
                ) : (
                    messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`message-bubble ${
                                msg.role === 'user' ? 'message-user' : 'message-assistant'
                            }`}
                        >
                            {msg.content}
                        </div>
                    ))
                )}

                <div ref={messagesEndRef} />
            </main>

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