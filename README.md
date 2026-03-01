# clickpass

Automated dev tool signup + API key extraction. Swipe right on tools, get accounts instantly.

## Quick Start

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
cd clickpass
uv sync

# 3. Run the server
uv run python server.py
```

Open **http://localhost:8000** in your browser.

## Pages

| URL | What it does |
|-----|-------------|
| `localhost:8000/` | Landing page |
| `localhost:8000/app.html` | Tinder-style swiper — pick 3 tools, get credentials + API keys instantly |
| `localhost:8000/dev.html` | Dev console — enter any URL, watch live signup in real-time |

## How it works

**app.html** — Pre-created accounts with real API keys for 10 dev tools (Klavis, Reducto, Helicone, Parea, Veryfi, DomeAPI, Sweep, Cactus Compute, AppSidekit, Daily). Swipe right on 3, get credentials + API keys + docs links.

**dev.html** — Enter any website URL, hit "clickpass it". The server spawns a cloud browser agent that:
1. Creates a disposable email (agentmail)
2. Opens the site in a cloud browser
3. Signs up with generated credentials
4. Handles email verification (HTTP or browser-based)
5. Logs in and extracts API keys
6. Streams everything back via SSE in real-time

## .env Keys Required

The `.env` file needs these for the signup engine (dev.html):

```
BROWSER_USE_API_KEY=...    # browser-use.com cloud browser
AGENTMAIL_API_KEY=...      # agentmail.to disposable emails
OPENAI_API_KEY=...         # for LLM (or use bu-2-0 model)
```

**app.html works without any API keys** — the accounts are pre-baked.

## Stack

- **Frontend**: Single-file HTML/CSS/JS, editorial aesthetic (Cormorant Garamond, gold accents, sharp edges)
- **Server**: FastAPI + uvicorn, SSE streaming
- **Signup engine**: browser-use (cloud), agentmail, faker
- **LLM**: bu-2-0 (Browser Use optimized model)

## File Structure

```
clickpass/
  server.py          # FastAPI server (serves HTML + /run SSE endpoint)
  app.html           # Tinder swiper with pre-created accounts
  dev.html           # Dev console for live signups
  landing.html       # Landing page
  index.html         # Redirect to landing
  pyproject.toml     # Python dependencies
  .env               # API keys
  v2/
    sigma_combined.py  # The signup engine
    pyproject.toml     # v2 dependencies
```
