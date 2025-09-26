# Imagen base oficial de Python
FROM python:3.11-slim

# Evitar problemas de buffer en logs
ENV PYTHONUNBUFFERED=1

# Carpeta de trabajo
WORKDIR /app

# Copiar requirements primero
COPY requirements.txt .

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Comando por defecto (se sobreescribe en docker-compose)
CMD ["python"]
