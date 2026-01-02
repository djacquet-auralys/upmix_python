// src/api/upmixClient.js
const API_BASE = import.meta.env.VITE_UPMIX_API_BASE || 'http://localhost:8000';

async function postJSON(path, body, { signal } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const upmixApi = {
  seatScore: (payload, opts) => postJSON('/api/upmix/v1/seat-score', payload, opts),
  placeSources: (payload, opts) => postJSON('/api/upmix/v1/place-sources', payload, opts),
  gainsDelays: (payload, opts) => postJSON('/api/upmix/v1/gains-delays', payload, opts),
  computeSeatScore: (payload, opts) => postJSON('/api/upmix/v1/compute-seat-score', payload, opts),
  sendUpmix: (payload, opts) => postJSON('/api/upmix/v1/send-upmix', payload, opts),
};
