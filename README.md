
# Advanced Machine Learning
This project was developed as part of the Advanced Topics in Machine Learning course, focusing on the design of AI-driven startups that combine technical feasibility with commercial viability.

- Simone Capata
- Giacomo Castiglioni
- Lucrezia Salerno

## API Key Setup

PharmaGuard AI uses the Groq API to power the LLM explanation layer 
(Stage 3 of the pipeline). In accordance with standard security practices 
for API credential management, the API key is not included in this 
repository. Each user is required to provision their own free Groq API key, 
which takes approximately two minutes and requires no credit card.

This approach reflects the same credential isolation principle that 
PharmaGuard AI enforces in its production architecture: sensitive 
configuration parameters are never stored in version-controlled codebases.

### Setup Instructions

1. Clone the repository
bashgit clone https://github.com/LucreziaSalerno/AdvancedML.git
cd AdvancedML

2. Copy the example environment file:
cp .env.example .env

3. Obtain a free Groq API key at https://console.groq.com  
   (Login → API Keys → Create API Key — no credit card required)

4. Open `.env` and replace the placeholder with your key:
GROQ_API_KEY=gsk_your_actual_key_here

5. Install the required dipendencies:
pip install -r requirements.txt

6. Run the application:
streamlit run app_real.py

PharmaGuard AI
AI-Powered Pharmaceutical Fraud Detection
Nova School of Business and Economics — 2758-T4 Advanced Topics in Machine Learning

Overview
PharmaGuard AI is a B2B SaaS prototype that detects fraudulent and off-label prescription patterns in real time. The platform combines a deterministic anomaly detection engine (Isolation Forest) with a regulatory knowledge base (RAG pipeline on EMA/FDA documents) and an LLM explanation layer (Llama 3.3 via Groq), delivered through a human-in-the-loop Streamlit dashboard.

Architecture
Raw prescription data
    → [Stage 1] DLP Pipeline         dlp_anonymisation.ipynb
        Regex + SHA-256 hashing + spaCy NER
    → Anonymised data
    → [Stage 2] Isolation Forest      isolation_forest.ipynb
        Anomaly detection on 200K CMS Medicare records
    → Anomaly scores
    → [Stage 2] RAG Pipeline          rag_pipeline.ipynb
        FAISS vector index on EMA/FDA regulatory documents
    → Regulatory references
    → [Stage 2] Kill Switch check
        If FAISS distance > 1.0 → block Groq, return manual review message
    → [Stage 2] Groq API              groq_explanations.ipynb
        Llama 3.3 70B generates plain-English compliance report
    → Pre-calculated explanations (pharmaguard_explained.csv)
    → [Frontend] Streamlit Dashboard  app.py
        Human-in-the-loop review workflow

Repository Structure
AdvancedML/
│
├── data/
│   ├── pharmaguard_clean.csv          # Cleaned CMS dataset (89,710 rows)
│   ├── pharmaguard_clean_anon.csv     # After DLP anonymisation
│   ├── pharmaguard_fraud.csv          # After synthetic fraud injection
│   ├── pharmaguard_results.csv        # After Isolation Forest scoring
│   └── pharmaguard_explained.csv      # After Groq explanations
│
├── rag_assets/
│   ├── ema_guideline.pdf              # EMA GVP Module VI Rev 2
│   ├── fda_offlabel.pdf               # FDA Off-Label Guidance
│   ├── fda_fraud.pdf                  # FDA Compliance Program 7363.001
│   ├── pharmaguard_index.faiss        # FAISS vector index
│   └── pharmaguard_chunks.pkl         # RAG document chunks
│
├── notebooks/
│   ├── 01_size_reduction_dataset.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_dlp_anonymisation.ipynb
│   ├── 04_fraud_injection.ipynb
│   ├── 05_isolation_forest.ipynb
│   ├── 06_rag_pipeline.ipynb
│   └── 07_groq_explanations.ipynb
│
├── app.py                             # Streamlit dashboard
├── requirements.txt
├── .env                               # API keys — NOT committed to git
├── .gitignore
└── README.md


Running the Full Pipeline
Run notebooks in order if you want to reproduce results from scratch:
StepNotebookOutput101_size_reduction_dataset.ipynbSampled CMS dataset (200K rows)202_data_cleaning.ipynbpharmaguard_clean.csv303_dlp_anonymisation.ipynbpharmaguard_clean_anon.csv404_fraud_injection.ipynbpharmaguard_fraud.csv505_isolation_forest.ipynbpharmaguard_results.csv606_rag_pipeline.ipynbpharmaguard_index.faiss + pharmaguard_chunks.pkl707_groq_explanations.ipynbpharmaguard_explained.csv

Steps 1–6 require no API keys. Step 7 requires a Groq API key in .env.


Dataset
CMS Medicare Part D Prescribers by Provider and Drug (2021)

Source: data.cms.gov
Original size: ~26 million rows
Prototype sample: 200,000 rows (stratified by specialty and drug)
Already fully anonymised at source — no patient names or direct identifiers

Synthetic fraud patterns injected during Step 4:

Volume fraud (~4% of records)
Cost fraud (~4% of records)
Off-label proxy (~4% of records)
Total flagged by Isolation Forest: ~12.5%


Regulatory Knowledge Base
Three official documents indexed in the RAG pipeline:
DocumentSourceEMA Guideline on Good Pharmacovigilance Practices (GVP Module VI, Rev 2)EMAFDA Guidance on Responding to Unsolicited Requests for Off-Label InformationFDAFDA Compliance Program Guidance Manual — Health Fraud (Program 7363.001)FDA

AI Safety Features

Kill Switch: if FAISS distance > 1.0, Groq is not called — dashboard returns "manual review required"
Separation of concerns: Isolation Forest makes all fraud flagging decisions; LLM only explains
Intentional friction: compliance officers must select violation type and write 10-word justification before recording a decision
Human-in-the-loop: no automated action — every flag requires human review
Model limitations disclosure: dashboard displays explicit notice that model was trained on US Medicare data


Tech Stack
ComponentTechnologyCostStage 1 DLPPython + SHA-256 + spaCyFreeAnomaly Detectionscikit-learn Isolation ForestFreeVector StoreFAISS (in-memory)FreeEmbeddingssentence-transformers all-MiniLM-L6-v2FreeLLM ExplanationsLlama 3.3 70B via Groq APIFree tierDashboardStreamlitFreePDF ParsingpypdfFree
Total infrastructure cost: €0

Academic Context
This prototype was built for course 2758-T4 Advanced Topics in Machine Learning at Nova School of Business and Economics, Lisbon, Portugal. All use of generative AI tools is documented in the GenAI Transparency Log in the accompanying business plan report.