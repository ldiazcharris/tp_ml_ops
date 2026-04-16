# Trabajo Práctico de Operaciones de Aprendizaje Automático

### Equipo

- Ronald Uthurralt
- Luis David Díaz Charris
- Juan Pablo Skobalski

### Caso de uso

Se crea una herramienta que sirva para medicos para poder predecir si 
una persona tiene riesgo de tener un ACV o no. El modelo puede ser utilizado para:

1. Clasificar nuevos pacientes según su probabilidad de stroke.
2. Priorizar casos para screening o seguimiento preventivo.
3. Asistir a equipos clínicos o de gestión con una señal temprana de riesgo.
4. Exponer esa predicción desde una API para integrarla con otros sistemas.

Desde el equipo de MLOps nos enfocamos en volver ese modelo reproducible,
versionable y consumible, entrenarlo de manera orquestada, registrar métricas y
artefactos, publicarlo en MLflow y servirlo a través de FastAPI y Streamlit.

El threshold seleccionado en la materia de Aprendizaje automatico, 
es simplemente figurativo y el que mejor resultado nos daba, de todas maneras
la herramienta estaría hecha para que el médico tome la decision.

### Relación con el TP de Aprendizaje Automático

Este repositorio toma como base el trabajo desarrollado en Aprendizaje Automatico, 
donde comparamos distintos modelos para la predicción de accidentes cerebro vasculares
y selecciona una versión final de `RandomForestClassifier` con ajuste de threshold.


### Dataset y objetivo del servicio

El proyecto utiliza el dataset clínico de stroke trabajado en Aprendizaje
Automático.

- Variables demográficas y clínicas del paciente.
- Target binario `stroke`.
- Dataset fuertemente desbalanceado.

El objetivo del servicio es recibir los atributos de un paciente y devolver:

- La probabilidad estimada de stroke.
- La clase final según el threshold operativo.
- La versión del modelo servida por la API.

El flujo está preparado para entrenar directamente con el CSV usado en
Aprendizaje Automático, ubicado en:

- `data/healthcare-dataset-stroke-data.csv`

Si ese archivo local no existe, el DAG puede usar como fallback una URL
configurada por `STROKE_DATASET_URL`.

### Arquitectura de alto nivel

```text
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
│  │     stroke_prediction_pipeline.py                  │       │
│  │     stroke_pipeline/*.py                           │       │
│  └────────────────────┬───────────────────────────────┘       │
│                       │ loguea experimentos                   │
│                       ▼                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  PostgreSQL │◄──│   MLflow    │──►│    MinIO    │          │
│  │  (MLflow)   │   │   :5001     │   │  (S3) :9001 │          │
│  └─────────────┘   └──────┬──────┘   └─────────────┘          │
│                            │                                  │
│                            │ expone artefactos                │
│                            ▼                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   Redis     │◄──│  FastAPI    │◄──│  Streamlit  │          │
│  │   :6379     │   │   :8800     │   │   :8501     │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| Airflow Webserver | [localhost:8080](http://localhost:8080) | UI para monitorear y disparar el pipeline |
| Airflow Scheduler | interno | Ejecuta las tareas del DAG |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos, registry y artefactos |
| MinIO | [localhost:9001](http://localhost:9001) | Consola del storage S3-compatible |
| FastAPI | [localhost:8800](http://localhost:8800) | API REST para inferencia del modelo |
| Streamlit | [localhost:8501](http://localhost:8501) | Interfaz para probar predicciones y ver métricas |
| Redis | `localhost:6379` | Cache de predicciones |
| PostgreSQL (MLflow) | interno | Backend store de MLflow |
| PostgreSQL (Airflow) | interno | Metadata store de Airflow |

### Flujo end-to-end

El pipeline `stroke_prediction_pipeline` ejecuta las siguientes etapas:

1. `ensure_artifact_bucket`
   Asegura que exista el bucket de MinIO usado para datasets y artefactos.
2. `get_data`
   Carga el CSV clínico desde `data/` y solo usa la URL configurada si no encuentra el archivo local.
3. `process_data`
   Elimina duplicados y normaliza columnas base para el entrenamiento.
4. `split_dataset`
   Genera conjuntos `train`, `validation` y `test` con partición estratificada.
5. `train_model`
   Entrena el pipeline final con preprocesamiento tabular, `SMOTE` y `RandomForestClassifier`.
6. `evaluate_model`
   Ajusta el threshold sobre validation y registra métricas finales sobre test.
7. `reload_prediction_api`
   Llama automáticamente al endpoint `/reload-model` para que la API quede sincronizada con el último modelo registrado.

### Estructura del proyecto

```text
tp_ml_ops/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── data/
│   └── .gitkeep
├── dockerfiles/
│   ├── Dockerfile.airflow
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.fastapi
│   └── Dockerfile.streamlit
├── dags/
│   ├── stroke_prediction_pipeline.py
│   └── stroke_pipeline/
│       ├── config.py
│       ├── data_tasks.py
│       ├── training_tasks.py
│       ├── evaluation_tasks.py
│       └── serving_tasks.py
├── api/
│   ├── api_config.py
│   ├── main.py
│   ├── model_service.py
│   ├── requirements.txt
│   └── schemas.py
├── streamlit/
│   ├── app.py
│   └── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

### Variables de entorno

Crear el archivo `.env` a partir del ejemplo:

```bash
cp .env.example .env
```

Variables requeridas:

- `PG_USER`
- `PG_PASSWORD`
- `PG_DATABASE`
- `AIRFLOW_DB_USER`
- `AIRFLOW_DB_PASSWORD`
- `AIRFLOW_DB_NAME`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MLFLOW_BUCKET_NAME`
- `AIRFLOW_SECRET_KEY`

Variable opcional:

- `STROKE_DATASET_URL`

### Inicio rápido

1. Clonar el repositorio:

   ```bash
   git clone <repo-url>
   cd tp_ml_ops
   ```

2. Crear `.env`:

   ```bash
   cp .env.example .env
   ```

3. Ubicar el CSV del TP de Aprendizaje Automático en:

   ```text
   data/healthcare-dataset-stroke-data.csv
   ```

   Ese es ahora el camino principal de entrenamiento.

4. Levantar la infraestructura:

   ```bash
   docker compose up --build -d
   ```

5. Verificar servicios:

   ```bash
   docker compose ps
   ```

6. Abrir Airflow en [http://localhost:8080](http://localhost:8080)

   Credenciales por defecto:

   - usuario: `admin`
   - contraseña: `admin`

7. Activar y ejecutar manualmente el DAG `stroke_prediction_pipeline`.

8. Esperar a que todas las tareas queden en verde.

   Cuando el DAG finaliza:

   - el bucket queda validado,
   - el dataset procesado queda versionado en MinIO,
   - el modelo final queda registrado en MLflow,
   - la API recarga automáticamente la última versión.

9. Consumir predicciones:

   - FastAPI docs: [http://localhost:8800/docs](http://localhost:8800/docs)
   - Streamlit: [http://localhost:8501](http://localhost:8501)

10. Ver métricas y runs:

   - MLflow: [http://localhost:5001](http://localhost:5001)

### ¿Qué se debería ver en MLflow?

- El experimento `stroke_prediction`.
- Al menos un run de entrenamiento.
- Parámetros del modelo y del threshold.
- Métricas de validation y test.
- Artefactos de evaluación.
- El modelo registrado `stroke_prediction_model`.

### Notas

- El bucket puede crearse durante el bootstrap por `create_s3_bucket`, pero el DAG también lo valida con `ensure_artifact_bucket`.
- El DAG principal quedó reducido a la orquestación, mientras que las tareas se separaron en módulos dentro de `dags/stroke_pipeline/`.
- Si existe una copia local del dataset en `data/`, el entrenamiento usa ese archivo como fuente principal.
- `STROKE_DATASET_URL` queda solamente como fallback para ejecuciones sin el CSV local.
- La API no requiere recarga manual después del entrenamiento: el DAG ejecuta `/reload-model` al finalizar.
- Streamlit muestra la fecha del último run en formato legible.
- El sistema despliega el modelo final seleccionado del TP, no toda la lógica exploratoria del notebook.

### Para detener el stack

```bash
docker compose down
```

Para eliminar también los volúmenes persistidos:

```bash
docker compose down -v
```
