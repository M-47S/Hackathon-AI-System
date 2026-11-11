import { useState, useRef, useEffect, type SyntheticEvent } from "react";
import type { ChatMessage } from "../types/api";
import './Chat.css'


export function Chat() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        if (textAreaRef.current){
            textAreaRef.current.style.height = 'auto';
            textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
        }
    }, [input]);

    function sendMessage() {
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
    
    function handleFormSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        sendMessage();
    }

    function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }


    return (
        <div className="chat-container">

            <main className="chat-messages">
                {messages.length === 0 ? (<p className="chat-empty">Начните диалог</p>) : (messages.map((msg) => (<div
                    key={msg.id}
                    className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                        {msg.content}
                </div>)))}
                <div ref={messagesEndRef} />
            </main>

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
                    disabled={!input.trim()}
                >
                    Отправить
                </button>
            </form>
        </div>
    );
}