import os
import tempfile

import pandas as pd
import pdfplumber
import streamlit as st

from pipeline.agent_pipeline import (
    AgentPipeline,
    AgentRoleConfig,
    ChunkConfig,
    ChunkStrategy,
    NERBackend,
    NERConfig,
    PipelineConfig,
    VLLMModelConfig,
)

# ---------------------------------------------------------------------------
# Module-level NER state (lazy-loaded on first use)
# ---------------------------------------------------------------------------

_spacy_nlp = None
NER_AVAILABLE = True

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_pipeline(role_configs: tuple, ner_config: NERConfig, chunk_config: ChunkConfig):
    """Rebuild only when config objects change. Tuple required for hashability."""
    config = PipelineConfig(
        team_name="Document Classification Team",
        roles=list(role_configs),
        ner=ner_config,
        chunking=chunk_config,
    )
    return AgentPipeline(config)


@st.cache_data(ttl=30)
def fetch_available_models(vllm_url: str) -> list[str]:
    """Query the vLLM /v1/models endpoint. Re-fetches at most every 30 seconds."""
    try:
        import requests
        base = vllm_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        response = requests.get(f"{base}/models", timeout=3)
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
        return models if models else ["gemma3:12b"]
    except Exception:
        return ["gemma3:12b"]


@st.cache_data
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".parquet", ".pq")):
        return pd.read_parquet(uploaded_file)
    st.error("Unsupported format. Please upload a CSV or Parquet file.")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        pages = []
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                pages.append(
                    f"--- Page {page_num} ---\n{text}"
                    if text
                    else f"--- Page {page_num} ---\n[No text extracted]"
                )
        os.unlink(tmp_path)
        return "\n\n".join(pages)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return None


def chunk_text(text: str, pipeline: AgentPipeline) -> list[str]:
    """Delegate chunking to the pipeline's Chunker."""
    return pipeline.chunker.chunk(text)


def extract_entities(text: str, model_name: str = "xx_ent_wiki_sm") -> list[dict]:
    global _spacy_nlp
    try:
        if _spacy_nlp is None or _spacy_nlp.meta["name"] != model_name.replace("/", "_"):
            import spacy
            _spacy_nlp = spacy.load(model_name)
        doc = _spacy_nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    except Exception as e:
        st.warning(f"NER failed: {e}")
        return []


def render_text_with_entities(text: str, entities: list[dict]) -> str:
    if not entities:
        return text
    colors = {
        "PERSON": "#FFE5B4", "ORG": "#B4D7FF", "GPE": "#B4FFB4",
        "DATE": "#FFB4B4", "MONEY": "#FFD4B4", "PERCENT": "#FFD4FF", "FACILITY": "#D4FFFF",
    }
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        color = colors.get(ent["label"], "#FFFFB4")
        s, e = ent["start"], ent["end"]
        tag = (
            f'<mark style="background-color:{color};padding:2px 4px;'
            f'border-radius:3px;" title="{ent["label"]}">{text[s:e]}</mark>'
        )
        text = text[:s] + tag + text[e:]
    return text


def build_configs(
    agent_configs: list[dict],        # list of {description, system_message, model_type, temperature}
    global_vllm_url: str,
    global_max_tokens: int,
    ner_backend: NERBackend,
    ner_model_name: str,
    chunk_strategy: ChunkStrategy,
    chunk_max_tokens: int,
    chunk_overlap: int,
) -> tuple:
    role_configs = tuple(
        AgentRoleConfig(
            description=a["description"],
            system_message=a["system_message"],
            model=VLLMModelConfig(
                model_type=a["model_type"],
                url=global_vllm_url,
                max_tokens=global_max_tokens,
                temperature=a["temperature"],
            ),
        )
        for a in agent_configs
    )
    ner_config = NERConfig(backend=ner_backend, model_name=ner_model_name)
    chunk_config = ChunkConfig(
        strategy=chunk_strategy,
        max_tokens=chunk_max_tokens,
        overlap_sentences=chunk_overlap,
    )
    return role_configs, ner_config, chunk_config


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Multi-Agent Classifier", layout="wide",)
st.title("RAG — Multi-Agent Document Classifier")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

DEFAULT_AGENTS = [
    {
        "description": "Classifies the document",
        "name": "Agent 1: Initial Classification",
        "system_message": "You are a document classification assistant. Analyze the given text and classify it into relevant categories or topics. Be precise and concise.",
        "model_type": "gemma3:12b",
        "temperature": 0.2,
    },
    {
        "description": "Validates classification consistency",
        "name": "Agent 2: Validation",
        "system_message": "You are a validator. Review the classification provided above. Check for accuracy and flag any concerns or ambiguities. Confirm or refine the classification.",
        "model_type": "gemma3:12b",
        "temperature": 0.0,
    },
    {
        "description": "Finalises with confidence score",
        "name": "Agent 3: Final Review & Confidence",
        "system_message": "You are a final reviewer. Based on the validation above, provide a final confidence score (0-100) for the classification and finalize the result.",
        "model_type": "gemma3:12b",
        "temperature": 0.0,
    },
]

if "agents" not in st.session_state:
    st.session_state.agents = [a.copy() for a in DEFAULT_AGENTS]

if "classifications" not in st.session_state:
    st.session_state.classifications = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")

# ── Global vLLM settings ──────────────────────────────────────────────────
st.sidebar.subheader("Global vLLM Settings")

global_vllm_url = st.sidebar.text_input("vLLM endpoint", value="http://localhost:8000/v1")
global_max_tokens = st.sidebar.number_input(
    "Max tokens (all agents)", min_value=256, max_value=8192, value=2000, step=256
)

available_models = fetch_available_models(global_vllm_url)

st.sidebar.divider()

# ── NER ───────────────────────────────────────────────────────────────────
st.sidebar.subheader("NER Model")

ner_backend_choice = st.sidebar.radio("Backend", ["spaCy", "HuggingFace"], horizontal=True)
ner_backend = NERBackend.SPACY if ner_backend_choice == "spaCy" else NERBackend.HUGGINGFACE
ner_model_defaults = {"spaCy": "xx_ent_wiki_sm", "HuggingFace": "dslim/bert-base-NER"}
ner_model_name = st.sidebar.text_input(
    "NER model name", value=ner_model_defaults[ner_backend_choice]
)

st.sidebar.divider()

# ── Chunking ──────────────────────────────────────────────────────────────
st.sidebar.subheader("Chunking")

chunk_strategy_choice = st.sidebar.radio("Strategy", ["Sentence", "Paragraph"], horizontal=True)
chunk_strategy = (
    ChunkStrategy.SENTENCE if chunk_strategy_choice == "Sentence" else ChunkStrategy.PARAGRAPH
)
chunk_max_tokens = st.sidebar.number_input(
    "Max tokens per chunk", min_value=64, max_value=2048, value=512, step=64
)
chunk_overlap = st.sidebar.number_input(
    "Sentence overlap",
    min_value=0, max_value=5, value=1,
    disabled=(chunk_strategy == ChunkStrategy.PARAGRAPH),
)

st.sidebar.divider()

# ── Agent configuration ───────────────────────────────────────────────────
st.sidebar.subheader("Agent Pipeline")

agents = st.session_state.agents

# Add / remove buttons
col_add, col_reset = st.sidebar.columns(2)
with col_add:
    if st.button("＋ Add agent", use_container_width=True):
        n = len(agents) + 1
        agents.append({
            "description": f"Agent {n}",
            "name": f"Agent {n}",
            "system_message": "You are a helpful assistant.",
            "model_type": available_models[0],
            "temperature": 0.0,
        })
        st.rerun()
with col_reset:
    if st.button("Reset", use_container_width=True):
        st.session_state.agents = [a.copy() for a in DEFAULT_AGENTS]
        st.rerun()

st.sidebar.markdown("---")

# Per-agent controls
for i, agent in enumerate(agents):
    with st.sidebar.expander(f"**{agent['name']}**", expanded=True):
        agent["name"] = st.text_input(
            "Name", value=agent["name"], key=f"name_{i}"
        )
        agent["description"] = agent["name"]
        agent["system_message"] = st.text_area(
            "System prompt", value=agent["system_message"], height=100, key=f"prompt_{i}"
        )
        agent["model_type"] = st.selectbox(
            "Model", available_models,
            index=available_models.index(agent["model_type"]) if agent["model_type"] in available_models else 0,
            key=f"model_{i}",
        )
        agent["temperature"] = st.slider(
            "Temperature", min_value=0.0, max_value=1.0,
            value=float(agent["temperature"]), step=0.05, key=f"temp_{i}"
        )
        if len(agents) > 1:
            if st.button("🗑 Remove", key=f"remove_{i}", use_container_width=True):
                agents.pop(i)
                st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Single Text / Document", "Bulk Classification"])

# ==================== TAB 1 ====================
with tab1:
    st.write("Paste text, or upload a PDF.")
    st.divider()

    input_method = st.radio("Input method:", ["Paste Text", "Upload PDF"], horizontal=True)
    text_input = ""

    if input_method == "Paste Text":
        text_input = st.text_area(
            "Text / document:", height=200, placeholder="Paste your text here…"
        )
    else:
        pdf_file = st.file_uploader("Upload a PDF:", type=["pdf"])
        if pdf_file is not None:
            with st.spinner("Extracting text from PDF…"):
                extracted = extract_text_from_pdf(pdf_file)
            if extracted:
                st.success("Text extracted.")
                text_input = st.text_area(
                    "Review / edit extracted text:",
                    value=extracted,
                    height=250,
                    key="pdf_text_input",
                )

    if text_input.strip():
        st.divider()
        st.subheader("Text Analysis")
        col1, col2 = st.columns(2)
        with col1:
            enable_chunking = st.checkbox("Semantic chunking", value=True)
        with col2:
            enable_ner = st.checkbox("Entity extraction (NER)", value=NER_AVAILABLE)

        if enable_chunking or enable_ner:
            # Build a temporary pipeline just for preprocessing preview
            role_configs, ner_config, chunk_config = build_configs(
                agents, global_vllm_url, global_max_tokens,
                ner_backend, ner_model_name,
                chunk_strategy, chunk_max_tokens, chunk_overlap,
            )
            preview_pipeline = get_pipeline(role_configs, ner_config, chunk_config)

        if enable_chunking:
            with st.spinner("Chunking…"):
                chunks = chunk_text(text_input, preview_pipeline)
            st.info(f"Split into {len(chunks)} chunks")
            with st.expander("View chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)
                    st.divider()

        if enable_ner:
            with st.spinner("Extracting entities…"):
                entities = extract_entities(text_input, model_name=ner_model_name)
            if entities:
                st.info(f"Found {len(entities)} entities")
                with st.expander("Entity table"):
                    st.dataframe(
                        pd.DataFrame(entities)[["text", "label"]], use_container_width=True
                    )
                with st.expander("Annotated text"):
                    st.markdown(
                        render_text_with_entities(text_input, entities), unsafe_allow_html=True
                    )
            else:
                st.info("No entities found.")

        st.divider()

    if st.button("Classify", type="primary", key="classify_single"):
        if text_input.strip():
            role_configs, ner_config, chunk_config = build_configs(
                agents, global_vllm_url, global_max_tokens,
                ner_backend, ner_model_name,
                chunk_strategy, chunk_max_tokens, chunk_overlap,
            )
            pipeline = get_pipeline(role_configs, ner_config, chunk_config)

            with st.spinner("Running multi-agent reasoning…"):
                result = pipeline.run(text_input)

            st.success("Done.")
            st.subheader("Input")
            st.write(
                result["original_prompt"][:500] + "…"
                if len(result["original_prompt"]) > 500
                else result["original_prompt"]
            )
            st.subheader("Agent Reasoning")
            for i, step in enumerate(result["reasoning_chain"], 1):
                with st.expander(f"**{step['agent']}**", expanded=(i == 1)):
                    st.write("**Input received:**")
                    st.text(
                        step["input"][:300] + "…" if len(step["input"]) > 300 else step["input"]
                    )
                    st.divider()
                    st.write("**Reasoning:**")
                    st.text(step["reasoning"])
            st.subheader("Final Result")
            st.success(result["final_result"])
        else:
            st.warning("Please enter some text or upload a PDF first.")

# ==================== TAB 2 ====================
with tab2:
    st.write("Upload a CSV or Parquet file and classify multiple rows.")
    st.divider()

    uploaded_file = st.file_uploader("CSV or Parquet file:", type=["csv", "parquet", "pq"])

    if uploaded_file is not None:
        df = load_file(uploaded_file)

        if df is not None:
            st.success(f"Loaded {len(df)} rows.")
            text_column = st.selectbox("Text column:", df.columns)

            st.subheader("Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Row Selection")
            col1, col2 = st.columns([2, 2])
            with col1:
                selection_strategy = st.radio(
                    "Scope:",
                    ["All rows", "First N rows", "Random N rows", "Specific rows"],
                )

            selected_indices = []
            if selection_strategy == "All rows":
                selected_indices = list(range(len(df)))
                st.info(f"All {len(df)} rows selected.")
            elif selection_strategy == "First N rows":
                with col2:
                    n_rows = st.number_input("N:", min_value=1, max_value=len(df), value=10)
                selected_indices = list(range(int(n_rows)))
                st.info(f"First {len(selected_indices)} rows.")
            elif selection_strategy == "Random N rows":
                with col2:
                    n_rows = st.number_input("N:", min_value=1, max_value=len(df), value=10)
                selected_indices = list(
                    df.sample(n=int(n_rows), random_state=42).index.sort_values()
                )
                st.info(f"{len(selected_indices)} random rows.")
            elif selection_strategy == "Specific rows":
                selected_indices = st.multiselect(
                    "Select rows:",
                    options=range(len(df)),
                    format_func=lambda i: f"Row {i}: {df.iloc[i][text_column][:60]}…",
                )
                if selected_indices:
                    st.info(f"{len(selected_indices)} rows selected.")

            if st.button("Start Classification", type="primary"):
                if selected_indices:
                    role_configs, ner_config, chunk_config = build_configs(
                        agents, global_vllm_url, global_max_tokens,
                        ner_backend, ner_model_name,
                        chunk_strategy, chunk_max_tokens, chunk_overlap,
                    )
                    pipeline = get_pipeline(role_configs, ner_config, chunk_config)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, row_num in enumerate(selected_indices):
                        text = df.iloc[row_num][text_column]
                        status_text.text(f"Row {row_num} ({idx + 1}/{len(selected_indices)})…")
                        with st.spinner(f"Processing row {row_num}…"):
                            result = pipeline.run(str(text), task_id=f"row-{row_num}")
                        st.session_state.classifications[row_num] = result
                        progress_bar.progress((idx + 1) / len(selected_indices))

                    status_text.empty()
                    st.success(f"Classified {len(selected_indices)} row(s).")
                else:
                    st.warning("No rows selected.")

            if st.session_state.classifications:
                st.subheader(f"Results ({len(st.session_state.classifications)} completed)")
                for row_num in sorted(st.session_state.classifications.keys()):
                    result = st.session_state.classifications[row_num]
                    preview = df.iloc[row_num][text_column]
                    with st.expander(
                        f"**Row {row_num}**: {str(preview)[:80]}…", expanded=False
                    ):
                        st.write("**Original text:**")
                        st.text(result["original_prompt"])
                        st.divider()
                        st.write("**Agent reasoning:**")
                        for step in result["reasoning_chain"]:
                            with st.expander(step["agent"], expanded=False):
                                st.text(step["reasoning"])
                        st.divider()
                        st.write("**Final classification:**")
                        st.success(result["final_result"])

                st.divider()
                if st.button("Export to CSV"):
                    export_df = pd.DataFrame([
                        {
                            "row_id": row_num,
                            "original_text": r["original_prompt"],
                            "final_classification": r["final_result"],
                        }
                        for row_num, r in st.session_state.classifications.items()
                    ])
                    st.download_button(
                        label="Download CSV",
                        data=export_df.to_csv(index=False),
                        file_name="classifications.csv",
                        mime="text/csv",
                    )