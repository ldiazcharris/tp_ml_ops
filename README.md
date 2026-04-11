# TP ML Ops — CEIA

Trabajo práctico para la materia Operaciones de Aprendizaje Automatico (ML Ops) de la Carrera de Especialización en Inteligencia Artificial (CEIA).

El proyecto simula la infraestructura interna de una empresa ficticia ("ML Models and something more Inc.") que ofrece modelos de ML. 
La plataforma utiliza Apache Airflow para orquestación de pipelines, MLflow para tracking de experimentos, MinIO como data lake (S3-compatible) y PostgreSQL como base de datos.

### Equipo
- Ronald Uthurralt
- Luis David Díaz Charris
- Juan Pablo Skobalski

## Arquitectura

```
┌───────────────────────────────────────────────────────────────┐
│                       docker-compose                          │
│                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐         │
│  │   Airflow   │   │   Airflow   │   │  PostgreSQL  │         │
│  │  Webserver  │   │  Scheduler  │   │  (Airflow)   │         │
│  │   :8080     │   │             │   │              │         │
│  └──────┬──────┘   └──────┬──────┘   └──────────────┘         │
│         │                 │                                   │
│         └────────┬────────┘                                   │
│                  │ ejecuta DAGs                               │
│                  ▼                                            │
│  ┌────────────────────────────────────────────────────┐       │
│  │                    dags/                           │       │
│  │       wine_quality_pipeline.py                     │       │
│  └────────────────────┬───────────────────────────────┘       │
│                       │ loguea experimentos                   │
│                       ▼                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  PostgreSQL │◄──│   MLflow    │──►│    MinIO    │          │
│  │  (MLflow)   │   │   :5001     │   │  (S3) :9001 │          │
│  └─────────────┘   └──────┬──────┘   └─────────────┘          │
│                            │                                  │
│                            │ carga modelo                     │
│                            ▼                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   Redis     │◄──│  FastAPI    │◄──│  Streamlit  │          │
│  │   :6379     │   │   :8800     │   │   :8501     │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Estructura del proyecto

```
tp_ml_ops/
├── docker-compose.yml          # Definición de todos los servicios
├── .env.example                # Template de variables de entorno
├── .gitignore
├── requirements.txt            # Dependencias Python (Airflow tasks)
├── dockerfiles/
│   ├── Dockerfile.airflow      # Imagen de Airflow + dependencias de ML
│   ├── Dockerfile.mlflow       # Servidor MLflow
│   ├── Dockerfile.fastapi      # API REST
│   └── Dockerfile.streamlit    # Interfaz de usuario
├── dags/
│   └── wine_quality_pipeline.py  # DAG del pipeline de ML
├── api/
│   ├── main.py                 # Endpoints FastAPI
│   └── requirements.txt
├── streamlit/
│   ├── app.py                  # UI Streamlit
│   └── requirements.txt
├── data/                       # Datos locales (no versionados)
└── .github/
    └── workflows/
        └── ci.yml              # CI/CD con GitHub Actions
```

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| Airflow Webserver | [localhost:8080](http://localhost:8080) | UI para visualizar y monitorear DAGs |
| Airflow Scheduler | interno | Programa y ejecuta las tareas según dependencias |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos, parámetros, métricas y artefactos |
| MinIO | [localhost:9001](http://localhost:9001) | Consola de administración del data lake (S3) |
| FastAPI | [localhost:8800](http://localhost:8800) | API REST para predicciones del modelo |
| Streamlit | [localhost:8501](http://localhost:8501) | Interfaz de usuario para predicciones |
| Redis | localhost:6379 | Cache de predicciones |
| PostgreSQL (MLflow) | interno | Backend store de MLflow |
| PostgreSQL (Airflow) | interno | Metadata de Airflow |

## Modelo

Se utiliza un RandomForestClassifier sobre el dataset Wine de scikit-learn (178 muestras, 13 features, 3 clases). El pipeline de entrenamiento se ejecuta como un DAG en Airflow con las siguientes tareas:

1. get_data — Descarga el dataset y lo sube a MinIO
2. process_data — Elimina duplicados y nulos
3. split_dataset — Separa en train (80%) y test (20%)
4. train_model — Entrena el modelo y lo registra en MLflow (parámetros, métricas, artefactos)
5. evaluate_model — Evalúa en test set y loguea métricas finales (accuracy, precision, recall, F1)

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

5. Ejecutar el pipeline de entrenamiento:
   - Ir a Airflow: http://localhost:8080 (usuario: `admin`, contraseña: `admin`)
   - Activar el DAG `wine_quality_pipeline`
   - Ejecutarlo manualmente con el botón "Trigger DAG"

6. Usar la API de predicciones:
   - Docs interactivos: http://localhost:8800/docs
   - O usar Streamlit: http://localhost:8501

7. Ver experimentos en MLflow:
   - http://localhost:5001

8. Para detener los servicios:
   ```bash
   docker compose down
   ```
   Para eliminar también los volúmenes (datos persistidos):
   ```bash
   docker compose down -v
   ```

## Estado del proyecto

### Entrega 1 (Clases 1–4)

- [x] Infraestructura dockerizada con Docker Compose
- [x] Apache Airflow configurado (webserver + scheduler + PostgreSQL)
- [x] MLflow configurado con backend en PostgreSQL y artefactos en MinIO (S3)
- [x] MinIO como data lake S3-compatible
- [x] DAG de pipeline de ML (get_data → process_data → split → train → evaluate)
- [x] Integración del DAG con MLflow para logging de experimentos

### Entrega final

- [x] API REST con FastAPI para servir predicciones del modelo
- [x] Cache de predicciones con Redis
- [x] Interfaz de usuario con Streamlit
- [x] CI/CD con GitHub Actions (lint + build)

## Conceptos aplicados

- Orquestación con Airflow: los pipelines de ML se definen como DAGs con tareas encadenadas y dependencias explícitas. Se usa `@task.virtualenv` para aislar dependencias de cada tarea.
- Tracking de experimentos con MLflow: cada ejecución registra parámetros, métricas y artefactos (modelo, scaler) de forma centralizada en el Model Registry.
- Data lake con MinIO: los datos y artefactos se almacenan en storage S3-compatible, separando almacenamiento de cómputo.
- Serving con FastAPI: el modelo se carga desde MLflow Registry y se expone vía API REST. Las predicciones se cachean en Redis para evitar cómputo redundante.
- Infraestructura como código: toda la infraestructura se define en `docker-compose.yml` y es reproducible con un solo comando.
- CI/CD: GitHub Actions ejecuta linting y build de imágenes Docker en cada push/PR.
