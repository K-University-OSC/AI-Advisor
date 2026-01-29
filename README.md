# Advisor OSC - Multi-LLM Chatbot

A comprehensive multi-LLM chatbot application with RAG (Retrieval-Augmented Generation) capabilities, supporting multiple AI models including OpenAI GPT-4, Google Gemini, Anthropic Claude, and local models via Ollama.

## Features

- **Multi-LLM Support**: Connect to multiple LLM providers (OpenAI, Google, Anthropic, Perplexity, Ollama)
- **RAG System**: Document upload and intelligent retrieval for context-aware responses
- **Agent Mode**: Automatic tool selection for web search, document analysis, and memory recall
- **Admin Dashboard**: User management, usage analytics, and cost monitoring
- **Real-time Streaming**: Server-Sent Events (SSE) for live response streaming
- **File Attachments**: Support for images, PDFs, and documents with AI vision capabilities
- **Memory System**: User-specific fact storage for personalized conversations

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.10+ (for local development)
- PostgreSQL 15+
- Qdrant (for vector storage)

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/advisor_osc.git
cd advisor_osc
```

### 2. Configure Environment

```bash
# Copy example environment file
cp backend/.env.example backend/.env

# Edit .env with your API keys
nano backend/.env
```

Required API keys:
- `OPENAI_API_KEY` - OpenAI API key
- `JWT_SECRET_KEY` - Secret key for JWT tokens
- `DATABASE_URL` - PostgreSQL connection string

Optional API keys:
- `GOOGLE_API_KEY` - Google Gemini API
- `CLAUDE_API_KEY` - Anthropic Claude API
- `PERPLEXITY_API_KEY` - Perplexity API for web search

### 3. Run with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

Access the application:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **Admin Dashboard**: http://localhost:8080/#/admin

### 4. Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---

## Architecture

```
advisor_osc/
├── backend/              # FastAPI backend
│   ├── routers/          # API endpoints
│   │   ├── auth.py       # Authentication (signup/login)
│   │   ├── chat.py       # Chat API (streaming)
│   │   ├── admin.py      # Admin dashboard API
│   │   └── rag_chat.py   # RAG-enabled chat
│   ├── services/         # Business logic
│   │   ├── agent_service.py
│   │   └── memory_service.py
│   ├── utils/            # Helper functions
│   ├── database/         # Database models
│   └── server.py         # FastAPI main app
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   │   ├── Admin/    # Admin dashboard
│   │   │   └── Auth/     # Login/signup forms
│   │   ├── api/          # API client functions
│   │   └── styles/       # CSS styles
│   └── public/
├── docker-compose.yml    # Docker services
└── README.md
```

---

## User Roles

| Role | Description |
|------|-------------|
| `user` | Regular user - can use chat, upload files |
| `admin` | Administrator - full access to dashboard, user management |

---

## API Endpoints

### Authentication
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/signup` | Register new user | - |
| POST | `/api/auth/login` | User login | - |
| GET | `/api/auth/me` | Get current user info | Required |

### Chat
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/chat/send` | Send message (SSE streaming) | Required |
| GET | `/api/chat/sessions` | List user sessions | Required |
| GET | `/api/chat/sessions/{id}` | Get session messages | Required |
| DELETE | `/api/chat/sessions/{id}` | Delete session | Required |
| POST | `/api/chat/upload` | Upload file attachment | Required |

### Admin (requires admin role)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/admin/dashboard` | Dashboard statistics | Admin |
| GET | `/api/admin/users` | List all users | Admin |
| POST | `/api/admin/users/{id}/suspend` | Suspend user | Admin |
| POST | `/api/admin/users/{id}/activate` | Activate user | Admin |
| GET | `/api/admin/dashboard/costs` | Cost analysis | Admin |
| GET | `/api/admin/dashboard/usage-patterns` | Usage patterns | Admin |

---

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 (localhost only) | PostgreSQL database |
| qdrant | 6333 (localhost only) | Vector database for RAG |
| redis | 6379 (localhost only) | Caching and rate limiting |
| backend | 8000 | FastAPI backend |
| frontend | 8080 | React frontend with Nginx |

> **Security Note**: All database ports are bound to `127.0.0.1` only (not exposed externally)

---

## Configuration

### LLM Models

Edit `frontend/src/Config.js` to configure available models:

```javascript
export const LLM_MODELS = {
    'gpt4o': { name: 'GPT-4o', provider: 'OpenAI' },
    'gpt4o-mini': { name: 'GPT-4o Mini', provider: 'OpenAI' },
    'gemini-flash': { name: 'Gemini Flash', provider: 'Google' },
    'claude-haiku': { name: 'Claude Haiku', provider: 'Anthropic' },
    // Add more models as needed
};
```

### Usage Limits

Edit `backend/utils/usage_limits.py` to configure default limits:

```python
DEFAULT_LIMITS = {
    "daily_messages": 100,
    "daily_tokens": 50000,
    "monthly_messages": 2000,
    "monthly_tokens": 1000000,
    # ...
}
```

---

## Environment Variables

```bash
# =============================================================================
# Database
# =============================================================================
DATABASE_URL=postgresql+asyncpg://advisor:your_password@localhost:5432/advisor_db

# =============================================================================
# LLM API Keys
# =============================================================================
OPENAI_API_KEY=sk-your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
CLAUDE_API_KEY=sk-ant-your-anthropic-api-key
PERPLEXITY_API_KEY=pplx-your-perplexity-api-key

# =============================================================================
# RAG / Document Processing
# =============================================================================
QDRANT_HOST=localhost
QDRANT_PORT=6333

# =============================================================================
# Authentication
# =============================================================================
JWT_SECRET_KEY=your-jwt-secret-key-change-this-in-production

# =============================================================================
# Server Configuration
# =============================================================================
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

---

## RAG System (Retrieval-Augmented Generation)

### Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  1. Query Expansion (Local)         │
│  - Synonym dictionary (80+ terms)   │
│  - Processing: 0.02~1.2ms           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. Hybrid Search                   │
│  ├─ Vector Search (Qdrant)          │
│  ├─ BM25 Keyword Search             │
│  └─ RRF Fusion (K=60)               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. BGE Reranking (Cross-Encoder)   │
│  - BAAI/bge-reranker-v2-m3          │
│  - CPU-based (no GPU required)      │
│  - Processing: ~100ms               │
└─────────────────────────────────────┘
    │
    ▼
Personalized Search Results (Top-K)
```

### Benchmark Results

| Metric | Value |
|--------|-------|
| NDCG@5 | 0.526 |
| MRR | 0.684 |
| Hit Rate@1 | 56.7% |
| Avg Latency | 427ms |

---

## Security Notes

- All database ports are bound to `127.0.0.1` only (not exposed externally)
- Never commit `.env` files with real credentials
- Change `JWT_SECRET_KEY` in production
- Use strong passwords for database

---

## Troubleshooting

### Database connection issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres
```

### Frontend not connecting to backend
```bash
# Check nginx configuration
docker-compose exec frontend cat /etc/nginx/conf.d/default.conf

# Verify backend is accessible
curl http://localhost:8000/api/health
```

### Check all services
```bash
# View status of all containers
docker-compose ps

# View all logs
docker-compose logs -f
```

---

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Style

- Backend: Follow PEP 8, use type hints
- Frontend: Follow ESLint configuration

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Version History

- **v1.0.0** (OSC Release) - Single-tenant architecture for public release
  - Removed multi-tenant support
  - Simplified admin dashboard
  - GitHub-ready with sensitive info removed
