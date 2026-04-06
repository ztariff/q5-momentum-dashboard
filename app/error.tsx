'use client';

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div style={{ padding: '40px', color: '#e2e8f0', backgroundColor: '#0a0e1a', minHeight: '100vh' }}>
      <h2 style={{ color: '#ef4444' }}>Dashboard Error</h2>
      <pre style={{ color: '#94a3b8', fontSize: '12px', whiteSpace: 'pre-wrap' }}>{error.message}</pre>
      <button onClick={reset} style={{ marginTop: '20px', padding: '8px 16px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
        Try Again
      </button>
    </div>
  );
}
