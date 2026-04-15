from typing import Literal

from pydantic import BaseModel, Field


class StrokeFeatures(BaseModel):

    gender: Literal["Male", "Female", "Other"] = Field(..., description="Sexo")
    age: float = Field(..., ge=0, le=120, description="Edad del paciente")
    hypertension: int = Field(..., ge=0, le=1, description="1 si tiene hipertensión")
    heart_disease: int = Field(..., ge=0, le=1, description="1 si tiene enfermedad cardíaca")
    ever_married: Literal["Yes", "No"] = Field(..., description="Si alguna vez estuvo casado/a")
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"] = Field(..., description="Tipo de trabajo")
    residence_type: Literal["Urban", "Rural"] = Field(..., description="Tipo de residencia")
    avg_glucose_level: float = Field(..., ge=0, description="Nivel promedio de glucosa")
    bmi: float | None = Field(None, ge=0, description="Indice de masa corporal")
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown",] = Field(..., description="Estado de consumo de tabaco")


class PredictionResponse(BaseModel):

    stroke_probability: float
    prediction: int
    prediction_label: str
    threshold: float
    model_version: str
    cached: bool = False
