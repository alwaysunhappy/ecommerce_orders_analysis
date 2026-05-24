PYTHON ?= py

.PHONY: install data all dash dashboard test clean

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.download_data

all:
	$(PYTHON) run_pipeline.py

dash:
	streamlit run dashboard/app.py

dashboard:
	streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for pattern in ('data/processed/*.sqlite', 'reports/tables/*.csv', 'reports/figures/*.png') for p in Path('.').glob(pattern)]"
