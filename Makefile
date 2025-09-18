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
process-upload:
	$(PY) pipelines/process_upload.py

azure-test:
	$(PY) -c "from src.azure_io import check_azure_connection; check_azure_connection()"

upload-processed:
	$(PY) -c "from src.azure_io import upload_file_to_processed; import sys; upload_file_to_processed(sys.argv)" $(FILE)

list-azure:
	$(PY) -c "from src.azure_io import list_process