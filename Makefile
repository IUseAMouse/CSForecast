.PHONY: help install install-dev download-data preprocess train-all train-lstm train-transformer evaluate test test-coverage format lint typecheck check-all clean

PYTHON := python
UV := uv
MODELS_DIR := data/models
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
SEQ := 90 ## Modify at will to match paper experimental setup
OUT := 30 ## Same

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: 
	$(UV) pip install -e .

install-dev: 
	$(UV) pip install -e ".[dev]"

download-data:
	$(PYTHON) scripts/download_data.py

preprocess:  
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) scripts/preprocess_data.py \
		--input $(DATA_RAW)/player_data_all.json \
		--output $(DATA_PROCESSED)/player_data_all_cleaned.json

train-all: 
	@mkdir -p $(MODELS_DIR)
	@echo "Training models for 90/30 configuration..."
	$(PYTHON) scripts/train_models.py --model all --seq-length 90 --out-length 30
	@echo "Training models for 90/60 configuration..."
	$(PYTHON) scripts/train_models.py --model all --seq-length 90 --out-length 60
	@echo "Training models for 180/120 configuration..."
	$(PYTHON) scripts/train_models.py --model all --seq-length 180 --out-length 120
	@echo "Training models for 225/150 configuration..."
	$(PYTHON) scripts/train_models.py --model all --seq-length 226 --out-length 150

train-lstm: 
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) scripts/train_models.py --model lstm \
		--seq-length $(or $(SEQ),90) \
		--out-length $(or $(OUT),30)

train-transformer:  
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) scripts/train_models.py --model transformer \
		--seq-length $(or $(SEQ),90) \
		--out-length $(or $(OUT),30)

train-ridge:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) scripts/train_models.py --model ridge \
		--seq-length $(or $(SEQ),90) \
		--out-length $(or $(OUT),30)

evaluate: 
	$(PYTHON) scripts/evaluate_models.py --models-dir $(MODELS_DIR) --plot

format:  
	black src/ scripts/ tests/

lint:  
	ruff check src/ scripts/ tests/

typecheck:  
	mypy src/ scripts/

check-all: format lint typecheck test  

clean: 
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

clean-data: 
	rm -rf $(DATA_PROCESSED)/*
	rm -rf $(MODELS_DIR)/*

clean-all: clean clean-data  

.DEFAULT_GOAL := help