from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Thesis NetMon API")

class PredictRequest(BaseModel):
    provider: str
    vm_size: str
    l4_protocol: str
    region: str
    az: str | None = None
    cpu_pct: float | None = None
    mem_pct: float | None = None
    packet_loss: float | None = None
    jitter_ms: float | None = None

@app.get("/health")
def health():
    return {"status": "ok"}
