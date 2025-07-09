# syntax=docker/dockerfile:1

ARG NODE_VERSION=22.13.1

# --- Build Stage ---
FROM node:${NODE_VERSION}-slim AS builder
WORKDIR /app

# Copy only package.json and package-lock.json for dependency install
COPY --link ../package.json ../package-lock.json ./

# Install dependencies with cache and bind mounts
RUN --mount=type=cache,target=/root/.npm \
    --mount=type=bind,source=../package.json,target=package.json \
    --mount=type=bind,source=../package-lock.json,target=package-lock.json \
    npm ci

# Copy the rest of the app source
COPY --link . .

# Build the app (Vite + TypeScript)
RUN --mount=type=cache,target=/root/.npm \
    npm run build

# Remove dev dependencies for production
RUN --mount=type=cache,target=/root/.npm \
    npm prune --production

# --- Production Stage ---
FROM node:${NODE_VERSION}-slim AS final
WORKDIR /app

# Create non-root user
RUN addgroup --system lexos && adduser --system --ingroup lexos lexos

# Copy built app and production deps from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=4096"
USER lexos

EXPOSE 4173 3000
CMD ["npx", "vite", "preview", "--host", "0.0.0.0"]
