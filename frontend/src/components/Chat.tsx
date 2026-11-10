import { useState, useRef, useEffect, type SyntheticEvent } from "react";
import type { ChatMessage } from "../types/api";
import './Chat.css'


export function Chat() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    function handleSend(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();

        if (!input.trim()) return;

        const newMessage: ChatMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            created_at: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, newMessage]);
        setInput('');
    }

    return (
        <div className="chat-container">
            <header className="chat-header">
                <h1>ENERGYTOOL AI</h1>
            </header>

            <main className="chat-messages">
                {messages.length === 0 ? (<p className="chat-empty">Начните диалог</p>) : (messages.map((msg) => (<div
                    key={msg.id}
                    className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                        {msg.content}
                </div>)))}
                <div ref={messagesEndRef} />
            </main>

            <form onSubmit={handleSend} className="chat-input-form">
                <input 
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Введите текст"
                    className="chat-input"
                />
                <button 
                    type="submit"
                    className="chat-submit"
                    disabled={!input.trim()}
                >
                    Отправить
                </button>
            </form>
        </div>
    );
}