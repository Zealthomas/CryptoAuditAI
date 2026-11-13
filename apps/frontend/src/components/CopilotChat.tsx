import React, { useState } from 'react';

type Props = {
  reportContext?: Record<string, unknown>;
  apiBaseUrl?: string;
};

type Message = {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
};

export default function CopilotChat({ reportContext = {}, apiBaseUrl = 'http://localhost:8000' }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    { id: 'welcome', sender: 'assistant', text: 'RiskLens copilot ready. Ask about wallet risk, scam signals, or contract behavior.', timestamp: new Date().toISOString() },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = { id: crypto.randomUUID(), sender: 'user', text: trimmed, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reportContext, query: trimmed }),
      });
      if (!response.ok) throw new Error('Copilot request failed');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Readable stream not available');

      const decoder = new TextDecoder();
      let assistantText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.replace(/^data:\s*/, '').trim();
          if (!payload || payload === '[DONE]') continue;
          try {
            const data = JSON.parse(payload);
            if (data.text) assistantText += data.text;
          } catch {
            // ignore non-JSON SSE chunks
          }
        }
      }

      setMessages((prev) => [...prev, { id: crypto.randomUUID(), sender: 'assistant', text: assistantText || 'No response returned.', timestamp: new Date().toISOString() }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Copilot request failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 16, background: '#111827', padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 12 }}>RiskLens Copilot</div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 260, overflowY: 'auto', marginBottom: 12 }}>
        {messages.map((msg) => (
          <div key={msg.id} style={{ alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start', background: msg.sender === 'user' ? '#2563eb' : '#1f2937', color: '#fff', padding: '8px 12px', borderRadius: 12, maxWidth: '80%' }}>
            {msg.text}
          </div>
        ))}
      </div>

      {error && <div style={{ color: '#fca5a5', marginBottom: 8 }}>{error}</div>}

      <div style={{ display: 'flex', gap: 8 }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about risk signals..." style={{ flex: 1, background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: 8, padding: '10px 12px' }} />
        <button type="button" onClick={sendMessage} disabled={isLoading} style={{ background: '#22c55e', color: '#111827', border: 'none', borderRadius: 8, padding: '10px 14px', cursor: isLoading ? 'not-allowed' : 'pointer' }}>
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
