# Drop-off Detective

Drop-off Detective is a demo analytics app for a Shopflo-like checkout product. It stitches session events, quantifies drop-offs, and generates root-cause narratives with experiment plans.

## One-command local run

```bash
./scripts/dev.sh
```

The script installs backend + frontend dependencies and starts FastAPI on `http://localhost:8000` and Next.js on `http://localhost:3000`. The backend auto-seeds ~5,000 sessions on startup.

## Manual setup (optional)

```bash
python -m pip install -r backend/requirements.txt
npm install
npm --prefix frontend install
npm run dev
```

## API highlights

- `POST /seed` — reseed ~5,000 sessions with realistic event sequences.
- `GET /metrics/overview` — conversion, abandonment, payment fail, prepaid share, and AOV by day.
- `GET /metrics/segments` — breakdowns by user type, device, payment method, gateway, cart value bucket, and pincode bucket.
- `GET /root_causes` — top 5 causes, anomalies, and logistic regression feature importance.

## Quick demo script

```bash
# 1) Start the app
./scripts/dev.sh

# 2) Reseed data (optional)
curl -X POST http://localhost:8000/seed

# 3) Open the UI
open http://localhost:3000
```

## Project structure

- `backend/` — FastAPI + SQLite + SQLAlchemy
- `frontend/` — Next.js + Tailwind dashboard UI
- `scripts/dev.sh` — one-command boot script
