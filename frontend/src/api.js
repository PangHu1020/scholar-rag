const BASE = '/api';

export async function fetchSessions() {
  const res = await fetch(`${BASE}/sessions`);
  return res.json();
}

export async function fetchHistory(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/history`);
  return res.json();
}

export async function deleteSession(sessionId) {
  const res = await fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' });
  return res.json();
}

export async function uploadFiles(files, onProgress) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const res = await fetch(`${BASE}/files/upload`, { method: 'POST', body: form });
  return res.json();
}

export async function fetchFiles() {
  const res = await fetch(`${BASE}/files`);
  return res.json();
}

export async function deleteFile(fileId) {
  const res = await fetch(`${BASE}/files/${fileId}`, { method: 'DELETE' });
  return res.json();
}

export async function clearCollection() {
  const res = await fetch(`${BASE}/collection`, { method: 'DELETE' });
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export function streamChat(query, sessionId, onEvent) {
  const ctrl = new AbortController();

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
    signal: ctrl.signal,
  }).then(async (res) => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        const raw = trimmed.slice(5).trim();
        if (raw === '[DONE]') continue;
        try {
          const evt = JSON.parse(raw);
          onEvent(evt);
        } catch {}
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onEvent({ type: 'error', data: err.message });
    }
  });

  return () => ctrl.abort();
}
