## Running the Project with Docker

This project is containerized using Docker and Docker Compose, with separate services for the Python backend and the TypeScript (Vite) frontend. Below are the project-specific instructions and requirements for running the application in Dockerized environments.

### Project-Specific Requirements

- **Python Backend**
  - Python version: **3.11** (as specified by `ARG PYTHON_VERSION=3.11`)
  - System dependencies: `build-essential`, `git`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `ffmpeg`, `libsndfile1`, `curl`
  - Runs as a non-root user `lexos`
  - Exposes ports: **8080** (API HTTP), **8081** (WebSocket)
  - Healthcheck endpoint: `http://localhost:8080/health`
  - Runtime directories created: `/app/lexos_memory`, `/app/logs`, `/app/static`, `/app/temp`, `/app/config`, `/app/uploads`, `/app/backups`

- **Frontend (TypeScript/Vite)**
  - Node.js version: **22.13.1** (as specified by `ARG NODE_VERSION=22.13.1`)
  - Runs as a non-root user `lexos`
  - Exposes ports: **4173** (Vite preview server), **3000** (optional/dev)

### Environment Variables

- No required environment variables are specified by default in the Dockerfiles or Compose file.
- If you need to provide environment variables, uncomment and use the `env_file` lines in the `docker-compose.yml` for each service.

### Build and Run Instructions

1. **Build and start all services:**
   ```sh
   docker compose up --build
   ```
   This will build both the backend and frontend images and start the containers.

2. **Accessing the services:**
   - **Backend API:** http://localhost:8080
   - **Backend WebSocket:** ws://localhost:8081
   - **Frontend (Vite preview):** http://localhost:4173

### Special Configuration

- The backend and frontend both run as a dedicated non-root user (`lexos`) for improved security.
- The backend creates several runtime directories for data, logs, uploads, etc., which can be mounted as volumes if persistent storage is needed.
- Healthchecks are configured for the backend service to ensure it is running and healthy.
- The frontend service depends on the backend and will start after the backend is healthy.

### Ports Exposed

| Service      | Container Port | Host Port | Purpose                |
|--------------|---------------|-----------|------------------------|
| python-app   | 8080          | 8080      | API HTTP               |
| python-app   | 8081          | 8081      | WebSocket              |
| ts-src       | 4173          | 4173      | Vite preview server    |
| ts-src       | 3000          | 3000      | (Optional/dev)         |

---

For any additional configuration (such as environment variables or volume mounts), edit the `docker-compose.yml` as needed for your deployment.