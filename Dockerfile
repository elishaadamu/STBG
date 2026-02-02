# Use a slim Python base image
FROM python:3.11-slim-bullseye

# Install system dependencies for GeoPandas, Fiona, Shapely, etc.
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libspatialindex-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency list first (better for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Render expects the service to listen on PORT
ENV PORT=8000

# Command to run your FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
