# LexOS - Advanced AI Operating System

A sophisticated AI-powered operating system with multi-agent orchestration, advanced memory management, and real-time monitoring capabilities.

## Quick Start

### Development Mode (Frontend Only)

If you want to run just the frontend for development and testing:

```bash
npm install
npm run dev
```

The frontend will run in development mode with mock data when the backend services are not available. This is perfect for UI development and testing.

### Full System with Backend Services

To run the complete LexOS system with all backend services:

```bash
# Install dependencies
make install

# Start all services with Docker
make docker
```

This will start:
- LexOS main service (port 8080)
- Ollama for AI models (port 11434)
- ChromaDB for vector storage (port 8000)
- Redis for caching (port 6379)
- PostgreSQL for data persistence (port 5432)
- Monitoring stack (Prometheus, Grafana)

### Available Commands

```bash
make help          # Show all available commands
make install       # Install Python dependencies
make setup         # Run complete setup
make run           # Start LexOS (Python only)
make test          # Run tests
make clean         # Clean temporary files
make docker        # Build and run with Docker
make docker-down   # Stop Docker containers
make logs          # Show Docker logs
make lint          # Lint code
make format        # Format code
make backup        # Backup memory and config
make restore       # Restore from backup
```

## Development Notes

### Frontend Development

The frontend is built with React + TypeScript + Vite and includes:
- Real-time WebSocket connections with fallback to mock data
- Responsive design with Tailwind CSS
- Advanced state management with React Context
- Multi-agent chat interface
- System monitoring dashboard
- Memory exploration tools

### Backend Services

The backend consists of multiple services orchestrated with Docker Compose:
- **LexOS Core**: Main AI orchestration service
- **Ollama**: Local AI model inference
- **ChromaDB**: Vector database for semantic memory
- **Redis**: Caching and message broker
- **PostgreSQL**: Persistent data storage

### Error Handling

The system is designed to gracefully handle backend unavailability:
- Frontend automatically switches to mock data mode when backend is offline
- WebSocket connections have timeout and retry logic
- API calls include proper error handling and fallbacks
- Development mode provides realistic mock responses

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   LexOS Core    │    │   AI Models     │
│   (React)       │◄──►│   (Python)      │◄──►│   (Ollama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   Vector DB     │              │
         └──────────────►│   (ChromaDB)    │◄─────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Cache/Queue   │
                        │   (Redis)       │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Database      │
                        │   (PostgreSQL)  │
                        └─────────────────┘
```

## Features

- **Multi-Agent Orchestration**: Coordinate multiple AI agents for complex tasks
- **Advanced Memory System**: Semantic memory with vector embeddings
- **Real-time Monitoring**: System metrics and performance dashboards
- **Model Management**: Load, unload, and switch between AI models
- **Chat Interface**: Interactive conversation with AI agents
- **Memory Explorer**: Browse and search through AI memories
- **System Settings**: Configure and customize the AI system

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Core Configuration
LEXOS_ENV=development
LEXOS_DEBUG=true
LEXOS_LOG_LEVEL=INFO

# Service Ports
LEXOS_PORT=8080
OLLAMA_PORT=11434
CHROMA_PORT=8000
REDIS_PORT=6379
POSTGRES_PORT=5432

# Database Configuration
POSTGRES_DB=lexos
POSTGRES_USER=lexos
POSTGRES_PASSWORD=lexos_secret

# Redis Configuration
REDIS_PASSWORD=lexos_secret

# GPU Configuration (if available)
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Backend Connection Issues

If you see connection errors to ports 8080 or 8081:

1. **Development Mode**: The frontend will automatically use mock data - this is normal
2. **Production Mode**: Ensure all Docker services are running with `make docker`
3. **Check Logs**: Use `make logs` to see service status
4. **Port Conflicts**: Ensure ports 8080, 8081, 11434, 8000, 6379, 5432 are available

### Docker Issues

If Docker commands fail:
- Ensure Docker and Docker Compose are installed
- Check if Docker daemon is running
- Verify sufficient disk space and memory
- For GPU support, ensure NVIDIA Docker runtime is installed

### Performance Issues

- Adjust memory limits in `docker-compose.yml`
- Configure GPU settings for your hardware
- Monitor resource usage with the built-in dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.