# Señales-de-Trading

Este repositorio contiene un aplicativo web que se despliega en el navegador para identificar oportunidades de entrada en trades.

Actualmente existen **tres páginas principales**:

- **Vision_General**: ofrece una vista global de los activos y tendencias.  
- **Vision_Individual**: permite detallar información específica de cada activo.  
- **Vision_Multiframe**: muestra análisis en distintos marcos de tiempo, útil para seguimiento y confirmación de señales.  

El flujo recomendado es utilizar **Vision_General** o **Vision_Multiframe** para generar una lista de seguimiento de activos y luego profundizar en cada uno con **Vision_Individual**.

---

## Configuración de entorno

El proyecto utiliza **Docker** para asegurar que todas las librerías y dependencias estén consistentes entre distintos sistemas.

- **Docker**: define un contenedor aislado que incluye todo el entorno de Python necesario para ejecutar la aplicación. Esto permite que la aplicación funcione de manera idéntica en cualquier máquina, sin conflictos de librerías o versiones.

- **Dockerfile**: especifica la imagen base de Python, instala todas las librerías necesarias y copia los archivos del proyecto dentro del contenedor. Actúa como la “receta” para construir la imagen del proyecto.

- **docker-compose.yml**: permite levantar simultáneamente las tres aplicaciones web en distintos puertos:
  - `http://localhost:8501/` → Vision_Individual  
  - `http://localhost:8502/` → Vision_General  
  - `http://localhost:8503/` → Vision_Multiframe  

- **requirements.txt**: lista todas las librerías de Python necesarias con sus versiones exactas, garantizando que el contenedor tenga un entorno reproducible y estable.

---

## Ejecución

Una vez configurado el entorno Docker, el proyecto se ejecuta de la siguiente manera:

1. **Construir la imagen Docker** (si es la primera vez o si se actualizó `requirements.txt`):

```bash
docker compose build
```
2. **Levantar las tres aplicaciones web** :

```bash
docker compose up
```

Para ejecutar en segundo plano
```bash
docker compose up -d
```
3. **Acceder a las paginas desde el navegador**:

  - Vision_Individual →   `http://localhost:8501/` 
  - Vision_General →   `http://localhost:8502/` 
  - Vision_Multiframe →   `http://localhost:8503/` 

4. **Detener las aplicaciones **
```bash
docker compose down
```

5. **Reinciar una app**
```bash
docker compose restart vision_multiframe
```
cambiar por la app que deseé.