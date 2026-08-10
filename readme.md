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