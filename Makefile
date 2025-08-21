# Makefile for the recommender pipeline + demo
# Usage examples:
#   make venv install              # set up env
#   make clean_data                # clean raw CSV -> cleaned_data.csv
#   make embed_products            # build product embeddings
#   make embed_customers           # build customer embeddings
#   make pipeline                  # clean_data -> embed_products -> embed_customers
#   make ui                        # run Streamlit demo


PY=python
PIP=pip

# -------- Config --------
RAW=data/data.csv
CLEAN=data/cleaned_data.csv

ART=artifacts
PMETA=$(ART)/products_meta.csv
PEMB=$(ART)/product_embeddings.npy
CMETA=$(ART)/customers_meta.csv
CEMB=$(ART)/customer_embeddings.npy

MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cpu        # set to cuda if you have a GPU

# -------- Environment --------
venv:
	$(PY) -m venv .venv

install:
	. .venv/bin/activate && $(PIP) install -r requirements.txt && \
	$(PIP) install -r requirements_api.txt || true

# -------- Pipeline steps --------
clean:
	rm -rf $(ART) || true
	mkdir -p $(ART) data

clean_data: $(CLEAN)

$(CLEAN): $(RAW) cleandata.py
	$(PY) cleandata.py --input $(RAW) --output $(CLEAN) --log-level INFO

embed_products: $(PMETA) $(PEMB)

$(PMETA) $(PEMB): $(CLEAN) embed_products.py
	$(PY) embed_products.py \
	  --input $(CLEAN) \
	  --meta-out $(PMETA) \
	  --emb-out $(PEMB) \
	  --model $(MODEL) \
	  --batch-size 512 --device $(DEVICE) --log-level INFO

embed_customers: $(CMETA) $(CEMB)

$(CMETA) $(CEMB): $(PMETA) $(PEMB) $(CLEAN) embed_customers.py
	$(PY) embed_customers.py \
	  --cleaned $(CLEAN) \
	  --products-meta $(PMETA) \
	  --product-emb $(PEMB) \
	  --customers-meta-out $(CMETA) \
	  --customer-emb-out $(CEMB) \
	  --tau-days 180 --min-products 1 --log-level INFO

# Run entire pipeline
pipeline: clean clean_data embed_products embed_customers

# -------- Demos --------
ui:
	streamlit run app.py


# -------- Quick test (smoke) --------
test:
	$(PY) -c "import pandas as pd; import numpy as np; print('OK: pandas', pd.__version__, 'numpy', np.__version__)"

.PHONY: venv install clean clean_data embed_products embed_customers pipeline ui api test
