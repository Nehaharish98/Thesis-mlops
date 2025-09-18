PY=python

install:
	 pip install -r requirements.txt

ingest:
	 $(PY) pipelines/ingest.py

features:
	 $(PY) pipelines/features.py

train:
	 $(PY) pipelines/training_flow.py

batch:
	 $(PY) pipelines/batch_inference.py

monitor:
	 $(PY) monitoring/drift_report.py

api:
	 uvicorn deployment.api.main:app --reload --port 8080

docker-build:
	 docker build -t thesis-netmon:latest .

test:
	 pytest -q
