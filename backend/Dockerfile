FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système nécessaires (notamment pour opencv-python, numpy, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer Gunicorn pour servir Django
CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]
