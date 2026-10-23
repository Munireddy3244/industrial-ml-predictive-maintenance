from pydantic import BaseModel, Field, ConfigDict

class SensorReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(min_length=1, max_length=64)
    temperature_c: float = Field(ge=-20, le=200, description="Sensor temperature in Celsius")
    vibration_mm_s: float = Field(ge=0, le=100, description="Vibration velocity in mm/s")
    speed_rpm: float = Field(ge=0, le=10000, description="Rotational speed in RPM")
    operating_hours: float = Field(ge=0, le=250000)

class PredictionResponse(BaseModel):
    equipment_id: str
    predicted_failure: bool
    failure_probability: float = Field(ge=0, le=1)
    risk_band: str
    threshold: float
    model_version: str
    disclaimer: str
