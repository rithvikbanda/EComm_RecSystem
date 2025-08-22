# 🛒 E-commerce Recommendation Engine
A recommendation system built end-to-end with **Python, FAISS, Sentence Transformers, and Streamlit**, based on invoice-based E-commerce data.

---

## Features
- **Data pipeline** → Cleans raw transactions into structured format  
- **Embeddings** → Semantic vectors for products & customers with Sentence Transformers  
- **Vector search** → FAISS index for fast similarity & product recommendations  
- **Advanced weighting** → Combines log-scaled quantity, recency decay, and inverse popularity for accurate recommendations  
- **Demo UI** → Interactive Streamlit app (text search, similar items, customer recs)  

---

## Tech Used
- **Python 3.11** – main language  
- **Pandas / NumPy** – data cleaning & aggregation  
- **Sentence Transformers (Hugging Face)** – semantic embeddings  
- **FAISS (Facebook AI Similarity Search)** – vector search & similarity  
- **Streamlit** – interactive demo UI  
- **Makefile** – reproducible pipeline orchestration  

---

## Project Structure
```text
data/                 # raw & cleaned CSVs
artifacts/            # generated embeddings & metadata
cleandata.py          # data cleaning pipeline
embed_products.py     # product embeddings
embed_customers.py    # customer embeddings
rag_search.py         # search + recommendation logic
app.py                # Streamlit demo UI
Makefile              # one-command pipeline 
requirements.txt      # UI + pipeline deps
```

---

## 🚀 Quickstart

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS/WSL
.venv\Scripts\activate        # Windows PowerShell

pip install -r requirements.txt
```

### 2. Run pipeline

**Linux/macOS/WSL**
```bash
make pipeline
```

**Windows (manual)**
```powershell
python cleandata.py --input data/data.csv --output data/cleaned_data.csv
python embed_products.py --input data/cleaned_data.csv --meta-out artifacts/products_meta.csv --emb-out artifacts/product_embeddings.npy
python embed_customers.py --cleaned data/cleaned_data.csv --products-meta artifacts/products_meta.csv --product-emb artifacts/product_embeddings.npy --customers-meta-out artifacts/customers_meta.csv --customer-emb-out artifacts/customer_embeddings.npy
```

### 3. Launch demo UI
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) to try:

- **Text search** → type queries like *“Party Balloons”*  
- **Similar items** → enter a product `StockCode`  
- **Customer recs** → enter a `CustomerID`, toggle *filter purchased* / *diversify*  

---

##  Architecture


