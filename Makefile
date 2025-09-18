PY=python

install:
\tpip install -r requirements.txt
\tpre-commit install || true

ingest:
\t$(PY) pipelines/ingest.py

features:
\t$(PY) pipelines/features.py

train:
\t$(PY) pipelines/training_flow.py

batch:
\t$(PY) pipelines/batch_inference.py

monitor:
\t$(PY) monitoring/drift_report.py

api:
\tuvicorn deployment.api.main:app --reload --port 8080

docker-build:
\tdocker build -t thesis-netmon:latest .

test:
\tpytest -q
