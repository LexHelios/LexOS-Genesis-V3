# syntax=docker/dockerfile:1

ARG NODE_VERSION=22.13.1

# --- Builder stage ---
FROM node:${NODE_VERSION}-slim AS builder
WORKDIR /app

# Install dependencies (npm ci for deterministic builds)
COPY --link package.json ./
# If you have a lock file, uncomment the next line and add it to .dockerignore
# COPY --link package-lock.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci

# Copy the rest of the application source
COPY --link . .

# Build the frontend (Vite + TypeScript)
RUN npm run build

# Remove dev dependencies to reduce image size
RUN npm prune --production

# --- Production stage ---
FROM node:${NODE_VERSION}-slim AS final
WORKDIR /app

# Create a non-root user
RUN addgroup --system lexos && adduser --system --ingroup lexos lexos

# Copy only the built app and production dependencies
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=4096"
USER lexos

EXPOSE 3000

CMD ["npx", "vite", "preview", "--port", "3000", "--host"]
