import React, { useEffect, useState } from 'react';

export type WatchlistItem = {
  id: string;
  target: string;
  chain: string;
  mode: 'token' | 'wallet' | 'contract';
  riskScore: number;
  previousScore?: number;
  riskLevel: string;
  updatedAt: string;
};

const STORAGE_KEY = 'risklens-watchlist';

export default function WatchlistDrawer() {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setItems(JSON.parse(stored));
      } catch {
        // ignore invalid data
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  }, [items]);

  const removeItem = (id: string) => {
    setItems((current) => current.filter((entry) => entry.id !== id));
  };

  return (
    <div style={{ position: 'relative' }}>
      <button type="button" onClick={() => setOpen((v) => !v)} style={{ background: '#111827', color: '#fff', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '8px 12px' }}>
        Watchlist
      </button>

      {open && (
        <div style={{ position: 'absolute', right: 0, top: 'calc(100% + 8px)', width: 320, background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 12, zIndex: 50 }}>
          {items.length === 0 ? (
            <div style={{ color: '#cbd5e1' }}>No items tracked.</div>
          ) : (
            items.map((item) => {
              const diff = typeof item.previousScore === 'number' ? item.riskScore - item.previousScore : 0;
              return (
                <div key={item.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <div>
                    <div style={{ fontWeight: 700 }}>{item.target}</div>
                    <div style={{ color: '#94a3b8', fontSize: 12 }}>{item.chain}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div>{item.riskScore}</div>
                    <div style={{ color: diff >= 0 ? '#fca5a5' : '#86efac', fontSize: 12 }}>
                      {diff >= 0 ? '+' : ''}{diff}
                    </div>
                  </div>
                  <button type="button" onClick={() => removeItem(item.id)} style={{ background: 'transparent', border: 'none', color: '#fca5a5', cursor: 'pointer' }}>
                    Remove
                  </button>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
