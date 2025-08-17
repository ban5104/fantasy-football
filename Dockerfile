# Build frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Python API with built frontend
FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn websockets

# Copy API code
COPY api/ ./api/
COPY data/ ./data/
COPY *.csv ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./static

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]