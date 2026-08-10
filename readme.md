


User clicks ticker
        ↓
GET /api/stock/{ticker}
        ↓
1. Load chart data          (existing)
2. Run research agent       (existing)
3. Run analytical engine    ← NEW (deterministic snapshot + charts)
        ↓
Return everything to frontend
        ↓
(Later) User asks question in sandbox
        ↓
POST /api/analyze  →  LLM receives the pre-built analytical picture as context

<img width="792" height="966" alt="Screenshot 2026-08-10 at 16 17 51" src="https://github.com/user-attachments/assets/29c016eb-6828-43c1-baf4-7aee51254598" />
