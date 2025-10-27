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

eda:
	$(PY) src/eda/network_eda.py

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# MLflow Commands
mlflow-start:
	python scripts/start_mlflow.py

mlflow-stop:
	python scripts/stop_mlflow.py

mlflow-dashboard:
	@echo "Opening MLflow dashboard..."
	@python -c "import webbrowser; webbrowser.open('http://127.0.0.1:5000')"

# Combined workflow commands
start-session: mlflow-start
	@echo "✅ Complete development session started!"

stop-session: mlflow-stop
	@echo "✅ Development session stopped!"

# ML training with tracking
train-with-tracking: mlflow-start
	python src/ml/train_models.py
	@echo "🎯 Training completed with MLflow tracking!"

# Reports
report-pdf:
	@echo "📄 Building PDF report from reports/network_monitoring_report.tex"
	@cd reports && (command -v tectonic >/dev/null 2>&1 \
		&& tectonic network_monitoring_report.tex \
		|| (command -v pdflatex >/dev/null 2>&1 \
			&& pdflatex -interaction=nonstopmode -halt-on-error network_monitoring_report.tex \
			|| (echo "No LaTeX engine found. Install 'tectonic' or 'pdflatex' and retry: make report-pdf" && exit 1)))

# Dataset building
build-ml-dataset:
	@echo "🧭 Building ML-friendly dataset from data/raw/PaperDataset"
	@python src/data/build_ml_dataset.py --input data/raw/PaperDataset --output-dir data/processed --format parquet || \
		( echo "Falling back to CSV (no parquet engine)" && \
		  python src/data/build_ml_dataset.py --input data/raw/PaperDataset --output-dir data/processed --format csv )
