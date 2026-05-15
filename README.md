# Patrick Carlberg – Data Science Learning Journey (2020–2025)

> **Strategic skill development period for deep technical expertise, project-based learning, and modern AI pipelines.**
<br>


## Executive Summary

This repository documents my systematic journey from domain expert in nanotechnology to full-stack, production-ready data scientist. Over five years, I completed 70+ hands-on projects, 50+ advanced certifications, and delivered modern machine learning solutions using the latest open-source libraries and cloud platforms.

- **Foundation Building:** Intensive upskilling in Python, statistics, ML, and cloud
- **Practical Application:** End-to-end projects from data wrangling to web deployment
- **Modern Workflows:** Productionization via Docker, FastAPI, LangChain, and more
- **Learning-in-Public:** Transparent record of raw code, refactoring, and skill development

All raw project code is documented *as is* to reflect genuine skill progression. Folders are organized to highlight learning phases and technology stacks, with honest badges indicating code maturity. [TIMELINE.md](./TIMELINE.md)
<br>
<br>


## Interactive Project Timeline

> 🚀 **[View Interactive Timeline](https://cjrockball.github.io/learning_journey/assets/timeline-complete.html)**
> 
> *Click above to see the full D3.js interactive visualization*
<br>



## Repository Structure

```shell
learning-journey/
│
├── README.md                    # Executive summary and navigation
├── TIMELINE.md                  # Chronological list of 70+ projects
├── GroupTimeline.md             # Projects sorted by library or subject 
├── coursera_certificates/       # Folder for certificate PDFs
├── 01_coursera_certificates/    # Sample of Coursera certificates 
├── 02_foundations/              # Phase 1 – Python, Stats, Early ML (2020–2021)
├── 03_machine_learning/         # Phase 2 – Core ML, XGB, PyTorch, TF (2021–2022)
├── 04_web_deployment/           # Phase 3 – Flask/FastAPI, APIs, web apps (2021–2023)
├── 05_advanced_computing/       # GPU/CUDA, Big Data, Optimization (2022–2024)
├── 06_quantitative_finance/     # Financial ML, RL, time series (2020–2024)
├── 07_kaggle_competitions/      # Competitive ML & public benchmarks (2021–2025)
├── 08_modern_ml_ai/             # Modern AI (LLMs, LangChain, RAG, NLP) (2024–2025)
├── 09_portfolio_showcase/       # Presentable, end-to-end project demos
```
<br>


## Phases of Development

**Phase 1: Foundations (2020–2021)**
- Upgraded Python, statistics, and data visualization skills
- Completed foundations via Coursera specializations

**Phase 2: Applied ML (2021–2022)**
- Built and iterated on Kaggle, UCI, and public datasets
- First production ML deployments

**Phase 3: Deployment + Performance (2021–2024)**
- Moved projects to web, experimented with containerization
- GPU, big-data, and scalable ML implementations

**Phase 4: Modern AI (2024–2025)**
- Integrated LLMs, LangChain, generative AI
- Competed in advanced Kaggle and NeurIPS challenges
<br>


## Featured Code Snippets

<details>
<summary><b>🐍 Phase 1: Early Python (2020)</b></summary>

```python
# First data visualization - humble beginnings
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stock_data.csv')
df['price'].plot(title='My First Stock Chart')
plt.show()
```
</details>

<details>
<summary><b>🤖 Phase 2: Machine Learning Pipeline (2021)</b></summary>

```python
# XGBoost model with proper validation
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```
</details>

<details>
<summary><b>🚀 Phase 3: Production FastAPI (2022-2023)</b></summary>

```python
# Full-stack ML serving with FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio

app = FastAPI(title="ML Model API", version="2.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "v1.2"

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model = await load_model_async(request.model_version)
        prediction = model.predict(torch.tensor(request.features))
        return {"prediction": prediction.item(), "confidence": 0.95}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
</details>

<details>
<summary><b>🧠 Phase 4: Modern AI Integration (2024-2025)</b></summary>

```python
# LangChain RAG system with custom retrieval
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

class CustomRAGSystem:
    def __init__(self, docs_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma.from_documents(
            documents=self.load_documents(docs_path),
            embedding=self.embeddings
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.setup_llm(),
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    async def query(self, question: str) -> str:
        return await self.qa_chain.arun(question)
```
</details>
<br>

## Raw/Unpolished Code & Growth Mindset

Some code in early folders is left intentionally raw or only lightly refactored. This is *deliberate*: it documents practical learning cycles, iterative improvements, and technological catch-up after a career pivot.

See [TIMELINE.md](./TIMELINE.md) for project-by-project progress, with major milestones and evolving code quality tagged along the way.
<br>
<br>


## Quick Links

- [Complete Project Timeline](./TIMELINE.md): All projects, with dates and themes
- [Group Timeline – by Tech/Library](./GroupTimeline.md)
- [Coursera Certificates (Summary)](https://github.com/CJRockball/learning_journey/tree/main/01_coursera_certificates)
- [Best Portfolio Projects](./08_portfolio_showcase/)
- [GitHub Profile](https://github.com/CJRockball)
<br>

## Value Proposition

- **Self-driven, systematic skill acquisition from first principles to production**
- **End-to-end project delivery, with honesty about unfinished/raw work**
- **Strong documentation practices, even for learning-phase code**

---

**Contact:** [Your email] • [LinkedIn] • [https://github.com/CJRockball](https://github.com/CJRockball)
