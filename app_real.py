import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
    }
    .stButton > button {
        border-radius: 8px;
        font-family: 'Open Sans', sans-serif;
        font-weight: 500;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ── Groq client ───────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# ── Load data and models ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("pharmaguard_results.csv")
    return df

@st.cache_resource
def load_rag():
    index = faiss.read_index("pharmaguard_index.faiss")
    with open("pharmaguard_chunks.pkl", "rb") as f:
        data = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return index, data['texts'], data['sources'], embedder

df = load_data()
index, texts, sources, embedder = load_rag()

# ── Core functions ────────────────────────────────────────────────────────────
def retrieve_regulation(query, top_k=1):
    q_emb = embedder.encode([query]).astype('float32')
    distances, indices = index.search(q_emb, top_k)
    idx = indices[0][0]
    return texts[idx][:400], sources[idx]

def generate_explanation(row, regulation, regulation_source):
    prompt = f"""You are a pharmaceutical compliance AI assistant.
Analyze the following suspicious prescription case and generate a clear,
concise explanation for a compliance officer.

CASE DETAILS:
- Drug: {row['generic_name']}
- Prescriber Specialty: {row['specialty']}
- Total Claims: {int(row['total_claims'])}
- Total Cost: ${float(row['total_cost']):,.2f}
- Total Patients: {int(row['total_patients'])}
- Anomaly Score: {float(row['anomaly_score']):.4f}
- Fraud Type Detected: {row['fraud_type']}

RELEVANT REGULATORY REFERENCE ({regulation_source}):
{regulation}

Generate a compliance report with:
1. A one-sentence summary of why this case is suspicious
2. The specific fraud pattern detected
3. The regulatory violation based on the reference above
4. Recommended action for the compliance officer

Keep it concise and professional. Maximum 150 words."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content

def get_severity(score):
    if score < -0.25:
        return "🔴 High"
    elif score < -0.10:
        return "🟡 Medium"
    else:
        return "🟢 Low"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ PharmaGuard AI")
st.sidebar.markdown("*AI-Powered Pharmaceutical Fraud Detection*")
st.sidebar.divider()

fraud_filter = st.sidebar.multiselect(
    "Filter by Fraud Type",
    options=["volume_fraud", "cost_fraud", "off_label"],
    default=["volume_fraud", "cost_fraud", "off_label"]
)

severity_filter = st.sidebar.selectbox(
    "Minimum Severity",
    options=["All", "Medium and above", "High only"]
)

st.sidebar.divider()
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Flagged Cases", f"{df['predicted_fraud'].sum():,}")
st.sidebar.metric("Fraud Rate", f"{df['predicted_fraud'].mean()*100:.1f}%")

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🛡️ PharmaGuard AI — Compliance Dashboard")
st.markdown("*Real-time pharmaceutical fraud detection powered by AI*")
st.divider()

# ── KPI metrics ───────────────────────────────────────────────────────────────
flagged = df[df['predicted_fraud'] == 1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Prescriptions Analysed", f"{len(df):,}")
with col2:
    st.metric("Anomalies Detected", f"{len(flagged):,}",
              delta=f"{len(flagged)/len(df)*100:.1f}% of total")
with col3:
    st.metric("High Severity Cases",
              f"{len(flagged[flagged['anomaly_score'] < -0.25]):,}")
with col4:
    st.metric("Est. Fraud Exposure",
              f"${flagged['total_cost'].sum()/1e6:.1f}M")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Fraud Distribution by Type")
    fraud_counts = flagged['fraud_type'].value_counts().reset_index()
    fraud_counts.columns = ['Fraud Type', 'Count']
    fig1 = px.bar(
        fraud_counts, x='Fraud Type', y='Count',
        color='Fraud Type',
        color_discrete_map={
            'cost_fraud': '#E74C3C',
            'volume_fraud': '#E67E22',
            'off_label': '#F39C12'
        }
    )
    fig1.update_layout(showlegend=False, height=300,
                       font=dict(family="Open Sans"))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("🗺️ Anomalies by State")
    state_counts = flagged['state'].value_counts().head(10).reset_index()
    state_counts.columns = ['State', 'Count']
    fig2 = px.bar(
        state_counts, x='Count', y='State',
        orientation='h', color='Count',
        color_continuous_scale='Reds'
    )
    fig2.update_layout(height=300, font=dict(family="Open Sans"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Flagged cases table ───────────────────────────────────────────────────────
st.subheader("🚨 Flagged Cases — Human Review Queue")

filtered = flagged[flagged['fraud_type'].isin(fraud_filter)].copy()
filtered['severity'] = filtered['anomaly_score'].apply(get_severity)

if severity_filter == "High only":
    filtered = filtered[filtered['anomaly_score'] < -0.25]
elif severity_filter == "Medium and above":
    filtered = filtered[filtered['anomaly_score'] < -0.10]

filtered = filtered.sort_values('anomaly_score', ascending=True).head(200)

st.caption(f"Showing {len(filtered):,} cases — sorted by anomaly score")

display_cols = ['prescriber_id', 'generic_name', 'specialty', 'state',
                'total_claims', 'total_cost', 'fraud_type', 'severity', 'anomaly_score']

st.dataframe(
    filtered[display_cols].rename(columns={
        'prescriber_id': 'Prescriber ID',
        'generic_name': 'Drug',
        'specialty': 'Specialty',
        'state': 'State',
        'total_claims': 'Total Claims',
        'total_cost': 'Total Cost ($)',
        'fraud_type': 'Fraud Type',
        'severity': 'Severity',
        'anomaly_score': 'Anomaly Score'
    }),
    use_container_width=True,
    height=400
)

st.divider()

# ── Case deep dive ────────────────────────────────────────────────────────────
st.subheader("🔍 Case Deep Dive — AI Explanation")

selected_id = st.selectbox(
    "Select a Prescriber ID to investigate:",
    options=filtered['prescriber_id'].astype(str).unique()
)

if selected_id:
    case = filtered[filtered['prescriber_id'].astype(str) == selected_id].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Drug", case['generic_name'])
        st.metric("Fraud Type", case['fraud_type'].replace('_', ' ').title())
    with col_b:
        st.metric("Total Claims", f"{int(case['total_claims']):,}")
        st.metric("Total Cost", f"${float(case['total_cost']):,.2f}")
    with col_c:
        st.metric("Total Patients", f"{int(case['total_patients']):,}")
        st.metric("Severity", get_severity(case['anomaly_score']))

    st.divider()

    if st.button("🤖 Generate AI Explanation", type="primary"):
        with st.spinner("Analyzing case and retrieving regulatory references..."):
            fraud_type = case['fraud_type']
            drug = case['generic_name']
            specialty = case['specialty']

            if fraud_type == 'volume_fraud':
                query = f"abnormal high prescription volume reporting {drug}"
            elif fraud_type == 'cost_fraud':
                query = f"fraud reimbursement cost pharmaceutical {drug}"
            else:
                query = f"off-label prescription {drug} {specialty}"

            regulation, reg_source = retrieve_regulation(query)
            explanation = generate_explanation(case, regulation, reg_source)

        st.success("Analysis complete!")

        st.markdown("#### 📋 AI Compliance Report")
        st.markdown(explanation)

        with st.expander("📖 Regulatory Reference Used"):
            st.markdown(f"**Source:** {reg_source}")
            st.markdown(f"*{regulation}*")

        st.divider()
        st.markdown("#### ⚖️ Compliance Officer Decision")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            if st.button("✅ Mark as Legitimate", use_container_width=True):
                st.success("Case marked as legitimate and closed.")
        with col_y:
            if st.button("⚠️ Escalate for Review", use_container_width=True):
                st.warning("Case escalated to senior compliance officer.")
        with col_z:
            if st.button("🚫 Flag as Confirmed Fraud", use_container_width=True):
                st.error("Case confirmed as fraud. Regulatory report initiated.")