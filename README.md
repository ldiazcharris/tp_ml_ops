# Trabajo Practico de Operaciones de Aprendizaje Automático I - 23Co2025 - CEIA - FIUBA

El proyecto simula la infraestructura interna de una empresa ficticia ("ML Models and something more Inc.") que ofrece modelos de ML. 
La plataforma utiliza Apache Airflow para orquestación de pipelines, MLflow para tracking de experimentos, MinIO como data lake (Simula un bucket de AWS S3) y PostgreSQL como base de datos.

### Equipo
- Ronald Uthurralt
- Luis David Díaz Charris
- Juan Pablo Skobalski

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│                    docker-compose                        │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Airflow   │    │   Airflow   │    │  PostgreSQL │   │
│  │  Webserver  │    │  Scheduler  │    │  (Airflow)  │   │
│  │   :8080     │    │             │    │             │   │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘   │
│         │                  │                             │
│         └─────────┬────────┘                             │
│                   │ ejecuta DAGs                         │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │                    dags/                           │  │
│  │            (pipelines de ML)                       │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │ loguea experimentos              │
│                       ▼                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  PostgreSQL │◄───│   MLflow    │───►│    MinIO    │   │
│  │  (MLflow)   │    │   :5001     │    │  (S3) :9001 │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Estructura del proyecto

```
tp_ml_ops/
├── docker-compose.yml        # Definición de todos los servicios
├── .env.example              # Template de variables de entorno
├── .gitignore
├── requirements.txt          # Dependencias Python (pandas, sklearn, mlflow, etc.)
├── dockerfiles/
│   ├── Dockerfile.airflow    # Imagen de Airflow + dependencias de ML
│   └── Dockerfile.mlflow     # Servidor MLflow + drivers de Postgres y S3
├── dags/                     # DAGs de Airflow (montado como volumen)
└── data/                     # Datos locales (no versionados)
```

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| Airflow Webserver | [localhost:8080](http://localhost:8080) | UI para visualizar y monitorear DAGs |
| Airflow Scheduler | interno | Programa y ejecuta las tareas según dependencias |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos, parámetros, métricas y artefactos |
| MinIO | [localhost:9001](http://localhost:9001) | Consola de administración del data lake (S3) |
| PostgreSQL (MLflow) | interno | Backend store de MLflow |
| PostgreSQL (Airflow) | interno | Metadata de Airflow |

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/)
- Git

## Inicio rápido

1. Clonar el repositorio:
   ```bash
   git clone <repo-url>
   cd tp_ml_ops
   ```

2. Crear el archivo de variables de entorno:
   ```bash
   cp .env.example .env
   ```

3. Levantar los servicios:
   ```bash
   docker compose up --build -d
   ```

4. Verificar que los servicios estén corriendo:
   ```bash
   docker compose ps
   ```

5. Acceder a las interfaces:
   - Airflow: http://localhost:8080 (usuario: `admin`, contraseña: `admin`)
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (usuario: `minio`, contraseña: `minio123`)

6. Para detener los servicios:
   ```bash
   docker compose down
   ```
   Para eliminar también los volúmenes (datos persistidos):
   ```bash
   docker compose down -v
   ```

## Estado actual

### Entrega 1 (Clases 1–4)

- [x] Infraestructura dockerizada con Docker Compose
- [x] Apache Airflow configurado (webserver + scheduler + PostgreSQL)
- [x] MLflow configurado con backend en PostgreSQL y artefactos en MinIO (S3)
- [x] MinIO como data lake S3-compatible
- [ ] DAG de pipeline de ML (get_data → process_data → split → train → evaluate)
- [ ] Integración del DAG con MLflow para logging de experimentos

### Entrega final — Pendiente

- [ ] API REST con FastAPI para servir predicciones del modelo
- [ ] Interfaz de usuario con Streamlit
- [ ] Redis para caching
- [ ] Monitoreo del modelo en producción
- [ ] CI/CD

## Conceptos aplicados

Este proyecto implementa prácticas de MLOps Nivel 1 (pipelines reproducibles):

- Orquestación con Airflow: los pipelines de ML se definen como DAGs, donde cada tarea tiene dependencias explícitas y se ejecuta de forma programada o por trigger manual.
- Tracking de experimentos con MLflow: cada ejecución del pipeline registra parámetros, métricas y artefactos (modelos serializados, scalers, etc.) de forma centralizada.
- Data lake con MinIO: los artefactos se almacenan en un storage S3-compatible, separando el almacenamiento de datos del cómputo.
- Infraestructura como código: toda la infraestructura se define en `docker-compose.yml` y es reproducible con un solo comando.
