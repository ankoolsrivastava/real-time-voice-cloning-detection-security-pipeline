# Frontend

React + TypeScript + Vite client for the real-time voice anti-spoofing backend.

## Run locally

```powershell
npm install
npm run dev
```

The client uses `http://127.0.0.1:8000/api` by default. Set `VITE_API_BASE_URL` when the backend runs at another address.

The browser microphone requires permission. The backend and its local ML artifacts must be running separately.
