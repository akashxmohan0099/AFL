# AFL Web App

Next.js frontend for the AFL prediction platform.

## Setup

Install dependencies:

```bash
npm ci
```

Create local env values from [`.env.local.example`](/Users/akash/Desktop/AFL/web/.env.local.example).

Supported env vars:

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | Browser API base URL. Defaults to `http://localhost:8000`. |
| `API_KEY` | Server-side API key for Next.js runtime requests. |
| `NEXT_PUBLIC_API_KEY` | Browser-visible shared API key. Only use this if key exposure is acceptable. |

## Commands

Development:

```bash
npm run dev
```

Production build:

```bash
npm run build
npm run start
```

Lint the frontend:

```bash
npm run lint
```

Smoke test the production app:

```bash
npm run build
npm run smoke
```

The smoke script starts `next start` on `127.0.0.1:3105` and verifies:

- `/`
- `/schedule`
- `/matches`
- `/players`
- `/odds`
- `/predictions`

## Docker

From the repo root:

```bash
docker compose up --build
```

The web container is built with Next.js standalone output and serves on `http://localhost:3000`.
