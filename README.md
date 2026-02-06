# PrismaChat

Chatbot inteligente con RAG (Retrieval Augmented Generation) que permite consultar documentos de la empresa usando un LLM open source corriendo localmente. Sin dependencias de APIs externas ni costos por uso.

## Características

- **LLM Local**: Usa Ollama con Llama 3.1 (8B) - sin enviar datos a terceros
- **RAG**: Busca en documentos indexados y genera respuestas contextuales
- **Multi-formato**: Soporta PDF, TXT, Markdown y DOCX
- **Streaming**: Respuestas en tiempo real via Server-Sent Events (SSE)
- **Sesiones**: Conversaciones con contexto persistente y TTL configurable
- **Cache**: Doble nivel (búsqueda + respuestas) con evicción LRU
- **Cola de inferencia**: Backpressure para manejar múltiples usuarios
- **Rate Limiting**: Token bucket por IP para proteger recursos
- **Logging**: Logs estructurados con Loguru, rotación diaria
- **API REST**: FastAPI con documentación automática (Swagger/OpenAPI)
- **Portable**: Docker-ready + configuración por variables de entorno

## Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| LLM | [Ollama](https://ollama.ai) + Llama 3.1 8B |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector Store | ChromaDB |
| Framework RAG | LangChain |
| API | FastAPI + Uvicorn |
| Logging | Loguru |
| Contenedor | Docker (opcional) |

## Arquitectura

```
PrismaChat/
├── app/
│   ├── main.py                 # Punto de entrada FastAPI
│   ├── api/
│   │   ├── dependencies.py     # Inyección de dependencias
│   │   └── routes/
│   │       ├── chat.py         # POST /api/v1/chat, /api/v1/chat/stream
│   │       ├── documents.py    # CRUD de documentos
│   │       ├── health.py       # Health checks
│   │       └── sessions.py     # Gestión de sesiones
│   ├── core/
│   │   ├── cache.py            # TTL Cache con LRU
│   │   ├── config.py           # Configuración centralizada
│   │   ├── exceptions.py       # Excepciones personalizadas
│   │   ├── logging.py          # Configuración de Loguru
│   │   ├── queue.py            # Cola async de inferencia
│   │   └── rate_limiter.py     # Token bucket rate limiter
│   ├── models/
│   │   ├── chat.py             # ChatRequest, ChatResponse
│   │   ├── document.py         # DocumentInfo, IngestResponse
│   │   └── session.py          # SessionData, Message
│   ├── repositories/
│   │   ├── session_store.py    # InMemory / File session store
│   │   └── vector_store.py     # ChromaDB + cache
│   └── services/
│       ├── chat_service.py     # Lógica RAG + cadena LLM
│       ├── document_service.py # Ingesta y chunking
│       └── session_service.py  # CRUD de sesiones
├── documents/                  # Documentos de la empresa
├── data/                       # ChromaDB + sesiones persistentes
├── logs/                       # Archivos de log
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Orquestación de servicios
├── requirements.txt            # Dependencias Python
└── .env.example                # Variables de entorno (plantilla)
```

**Flujo de datos:**

```
Cliente → API Routes → Services → Repositories → ChromaDB / Ollama
              ↓
        Session Store ← Contexto de conversación
              ↓
          Logging → Registro de operaciones
```

## Requisitos Previos

- **Python 3.11+** (para ejecución directa) o **Docker** (para contenedores)
- **Ollama** instalado con los modelos descargados
- **16GB RAM** mínimo recomendado

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/jairoortiz19/PrismaChat.git
cd PrismaChat
```

### 2. Instalar Ollama y modelos

```bash
# Instalar Ollama: https://ollama.ai/download
# Luego descargar los modelos:
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env según tu configuración
```

### 4a. Ejecución directa (sin Docker)

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate
# Activar (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar la API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4b. Ejecución con Docker

```bash
docker-compose up -d
```

> **Nota:** Si Ollama corre en el host (no en Docker), el `docker-compose.yml` ya está configurado para conectarse via `host.docker.internal`.

### 5. Verificar instalación

```bash
curl http://localhost:8000/health/live
# Respuesta esperada: {"status":"alive"}
```

## Uso

### Flujo típico

```bash
# 1. Subir documentos a la carpeta ./documents/

# 2. Ingestar documentos
curl -X POST http://localhost:8000/api/v1/documents/ingest

# 3. Crear sesión
curl -X POST http://localhost:8000/api/v1/sessions

# 4. Hacer preguntas (reemplazar SESSION_ID)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la política de vacaciones?", "session_id": "SESSION_ID"}'
```

### Subir archivos via API

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@mi_documento.pdf"
```

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado completo del sistema |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `POST` | `/api/v1/chat` | Enviar pregunta |
| `POST` | `/api/v1/chat/stream` | Chat con streaming (SSE) |
| `POST` | `/api/v1/documents/upload` | Subir documento |
| `POST` | `/api/v1/documents/ingest` | Procesar documentos |
| `GET` | `/api/v1/documents` | Listar documentos indexados |
| `DELETE` | `/api/v1/documents/{id}` | Eliminar documento |
| `POST` | `/api/v1/sessions` | Crear sesión |
| `GET` | `/api/v1/sessions/{id}` | Obtener sesión |
| `GET` | `/api/v1/sessions/{id}/messages` | Historial de mensajes |
| `DELETE` | `/api/v1/sessions/{id}` | Eliminar sesión |
| `POST` | `/api/v1/sessions/cleanup` | Limpiar sesiones expiradas |

La documentación interactiva (Swagger) está disponible en `http://localhost:8000/docs`.

## Configuración

Todas las opciones se controlan via variables de entorno (archivo `.env`):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL de Ollama |
| `LLM_MODEL` | `llama3.1:8b` | Modelo LLM a usar |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modelo de embeddings |
| `CHUNK_SIZE` | `1000` | Tamaño de chunks para documentos |
| `CHUNK_OVERLAP` | `200` | Solapamiento entre chunks |
| `RETRIEVER_K` | `4` | Documentos a recuperar por query |
| `SESSION_TTL_HOURS` | `24` | Tiempo de vida de sesiones |
| `SESSION_BACKEND` | `memory` | Backend: `memory` o `redis` |
| `CACHE_SEARCH_TTL` | `1800` | TTL cache de búsqueda (seg) |
| `CACHE_RESPONSE_TTL` | `3600` | TTL cache de respuestas (seg) |
| `QUEUE_MAX_CONCURRENT` | `2` | Workers de inferencia simultáneos |
| `QUEUE_MAX_SIZE` | `50` | Máximo de requests en cola |
| `RATE_LIMIT_CHAT_TOKENS` | `10` | Burst máximo para chat |
| `RATE_LIMIT_CHAT_REFILL` | `0.5` | Tokens/seg de recarga (chat) |
| `WORKERS` | `1` | Workers de Uvicorn |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

## Escalar el Sistema

### Más RAM / CPU

```env
# Usar un modelo más grande (requiere más RAM)
LLM_MODEL=llama3.1:70b

# Más workers de inferencia
QUEUE_MAX_CONCURRENT=4

# Más workers de Uvicorn
WORKERS=4
```

### Con GPU (NVIDIA)

Ollama detecta y usa GPU automáticamente. Para Docker, descomentar la sección de GPU en `docker-compose.yml`.

### Redis para sesiones (producción)

```env
SESSION_BACKEND=redis
REDIS_URL=redis://localhost:6379
```

Descomentar el servicio Redis en `docker-compose.yml`.

## Postman

Se incluye una colección de Postman con todos los endpoints preconfigurados:

**Archivo:** `Chatbot_RAG_API.postman_collection.json`

Importar en Postman y ajustar la variable `base_url` a tu servidor.

## Migrar a otro servidor

### Con Docker
```bash
# En el servidor destino:
git clone https://github.com/jairoortiz19/PrismaChat.git
cd PrismaChat
cp .env.example .env
# Ajustar .env
docker-compose up -d
```

### Sin Docker
```bash
git clone https://github.com/jairoortiz19/PrismaChat.git
cd PrismaChat
cp .env.example .env

# Instalar Ollama + modelos
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Instalar Python y dependencias
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt

# Iniciar
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Formatos Soportados

| Formato | Extensiones |
|---------|-------------|
| PDF | `.pdf` |
| Texto plano | `.txt` |
| Markdown | `.md` |
| Word | `.docx` |

## Licencia

MIT
