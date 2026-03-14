# Blurt

**An AI second brain that catches everything rattling around in your head and quietly makes sense of it for you.**

Blurt is a conversational AI companion with full-duplex voice interaction. You speak naturally, Blurt captures everything, silently classifies it, extracts entities, detects emotion, builds a compounding personal knowledge graph, syncs actions to your existing tools, and surfaces the right task at the right time.

## How It Works

```
Voice/Text In -> Classify -> Extract Entities -> Detect Emotion -> Store in Memory -> Surface Tasks
                    |               |                  |                  |
                 7 intents    people, places,    Plutchik 8 emotions   3-tier memory
                              projects, orgs     with intensity        (working, episodic, semantic)
```

### The Core Loop

1. **Dump** -- Say it or type it. No structure required.
2. **Classify** -- AI identifies intent: task, event, reminder, idea, journal entry, update, or question.
3. **Connect** -- AI extracts people, places, projects. Links them to what it already knows. Detects emotional state.
4. **Surface** -- When you need something to do, Blurt picks the right thing for your energy, mood, and schedule.
5. **Learn** -- Every interaction feeds memory. Completed tasks, skipped tasks, mood patterns -- it all compounds.

## Features

### AI Pipeline
- **Full-duplex voice input** via Gemini 2 multimodal API (raw audio in, understanding out)
- **7-intent silent classifier** with >85% confidence (task, event, reminder, idea, journal, update, question)
- **Entity extraction** -- people, places, projects, organizations from natural speech
- **Emotion detection** -- Plutchik's 8 primary emotions with intensity scoring and valence/arousal dimensions
- **Two-model strategy** -- Flash-Lite for classification/extraction (cheap, fast), Flash for reasoning/insights (smarter)

### Memory System (The Moat)
- **Working Memory** -- Session-scoped context with TTL expiration
- **Episodic Memory** -- Append-only observations, never deleted, compressed over time
- **Semantic Memory** -- Entity graph with relationship edges, co-mention strength, and vector embeddings via Gemini 2
- **Memory Promotion Pipeline** -- Automatic working -> episodic -> semantic promotion based on importance scoring

### Task Surfacing
- **21-factor scoring engine** -- mood, energy, time-of-day, calendar availability, entity relevance, behavioral signals
- **Thompson Sampling** -- Beta distribution-based behavioral learning that adapts to your patterns
- **Pattern detection** -- Learns rhythms like Thursday afternoon crashes and pre-10AM creativity
- **Anti-shame design** -- No streaks, no guilt, no overdue counters. "You're clear" is a valid state.

### Integrations
- **Google Calendar** -- Bidirectional sync (create, read, update events)
- **Notion** -- Sync tasks and notes
- **Local-only mode** -- Full feature parity with zero data leakage (socket-level egress guard)

### Security
- **AES-256-GCM** authenticated encryption for all data at rest
- **PBKDF2-HMAC-SHA256** key derivation (600,000 iterations, OWASP 2024)
- **Envelope encryption** -- master key protects data encryption keys
- **Egress guard** -- Blocks all outbound network requests in local-only mode

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone
git clone https://github.com/syarlag31/blurt.git
cd blurt

# Create virtual environment
uv venv --python 3.14
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini 2 API key ([get one](https://aistudio.google.com/apikey)) |
| `BLURT_MASTER_KEY` | Yes | Encryption master key (64 hex chars) |
| `BLURT_GOOGLE_CLIENT_ID` | For calendar sync | Google OAuth 2.0 client ID |
| `BLURT_GOOGLE_CLIENT_SECRET` | For calendar sync | Google OAuth 2.0 client secret |
| `NOTION_API_TOKEN` | For Notion sync | Notion API integration token |
| `BLURT_MODE` | No | `cloud` (default) or `local` |

Generate a master key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Run the Server

```bash
.venv/bin/uvicorn blurt.core.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Unit tests (no API key needed)
.venv/bin/pytest tests/ -v --ignore=tests/e2e_real

# Integration tests (no API key needed)
.venv/bin/pytest tests/e2e/ -v

# Real E2E tests (requires GEMINI_API_KEY in .env)
.venv/bin/pytest tests/e2e_real/ -v -m real_e2e
```

## Project Structure

```
blurt/
  api/              REST API endpoints (capture, calendar, patterns, recall, tasks)
  audio/            Full-duplex voice (WebSocket handler, chunker, pipeline)
  classification/   7-intent classifier (Flash-Lite, two-tier confidence)
  clients/          Gemini API clients (generation, embeddings, local fallback)
  config/           Settings and configuration
  core/             App factory, encryption, entity extraction, memory models
  extraction/       Entity extraction service
  gemini/           Gemini audio client
  integrations/     Google Calendar (OAuth, sync, polling) and Notion
  local/            Offline adapters (classifier, extractor, storage)
  memory/           3-tier memory (working, episodic, semantic, graph store)
  middleware/       Egress guard for local-only mode
  models/           Domain models (audio, emotions, entities, intents, sync)
  services/         Business logic (surfacing, Thompson Sampling, patterns, recall)
  sync/             Calendar sync orchestration and conflict resolution

tests/
  Unit tests        2,710 tests, 90% coverage
  tests/e2e/        355 integration tests (mocked external APIs)
  tests/e2e_real/   31 true E2E tests (real Gemini API calls)
```

## Design Principles

1. **Zero friction** -- Just dump. AI classifies.
2. **One task at a time** -- Never overwhelm.
3. **Anti-streak** -- Celebrate wins, never punish absence.
4. **Shame-proof** -- No overdue counters, no guilt language.
5. **Burst-friendly** -- 30-second sessions, not 30-minute planning.
6. **AI-first** -- Single input, AI routes everything.
7. **Memory compounds** -- Every interaction makes Blurt smarter.
8. **Never fabricate** -- If Blurt doesn't know, it says so.

## Built With

- **[Gemini 2](https://ai.google.dev/)** -- Multimodal voice, classification, embeddings
- **[FastAPI](https://fastapi.tiangolo.com/)** -- Async Python web framework
- **[Pydantic](https://docs.pydantic.dev/)** -- Data validation and settings
- **[Ouroboros](https://github.com/Q00/ouroboros)** -- Specification-first AI development harness used to generate and validate the initial backend implementation

## Development Harness

The initial backend was built using [Ouroboros](https://github.com/Q00/ouroboros), a specification-first AI development workflow engine. The process:

1. **Interview** -- Socratic questioning to crystallize the vision from a raw idea doc
2. **Seed** -- Generated a formal YAML specification with 17 acceptance criteria
3. **Execute** -- Ouroboros orchestrated parallel code generation across all ACs
4. **Evaluate** -- 3-stage verification pipeline (mechanical, semantic, consensus)
5. **Ralph** -- Automated fix loop until all mechanical checks passed (lint, build, types, tests)

The seed specification is preserved in [`seed.yaml`](seed.yaml).

## License

MIT
