# Makefile
# Purpose: Define common commands for the project

setup:
	./setup.sh

test:
	pytest tests/unit/ -v

run:
	python src/model.py
	python src/inference.py

app:
	streamlit run app.py

