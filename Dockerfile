# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn using the venv's Python directly
CMD ["uv", "run", "uvicorn", "src.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]