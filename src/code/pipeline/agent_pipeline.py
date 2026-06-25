from dataclasses import dataclass, field
from enum import Enum
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage


# ---------------------------------------------------------------------------
# Enums (set names for NER backend and chunking strategy)
# ---------------------------------------------------------------------------

class NERBackend(str, Enum):
    SPACY = "spacy"
    HUGGINGFACE = "huggingface"


class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VLLMModelConfig:
    model_type: str = "gemma3:12b"
    url: str = "http://localhost:8000/v1"
    temperature: float = 0.2
    max_tokens: int = 2000


@dataclass
class NERConfig:
    backend: NERBackend = NERBackend.SPACY
    model_name: str = "xx_ent_wiki_sm"
    device: str = "cuda"      # "cpu" | "cuda" | "cuda:0"
    batch_size: int = 32      # HuggingFace only


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    max_tokens: int = 512
    overlap_sentences: int = 1    # sentences shared between consecutive chunks
    paragraph_sep: str = "\n\n"   # delimiter used when strategy=PARAGRAPH


@dataclass
class AgentRoleConfig:
    description: str
    system_message: str
    model: VLLMModelConfig = field(default_factory=VLLMModelConfig)


@dataclass
class PipelineConfig:
    team_name: str = "Generic Task Team"
    roles: list[AgentRoleConfig] = field(default_factory=list)
    ner: NERConfig = field(default_factory=NERConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)


# ---------------------------------------------------------------------------
# NER runner
# ---------------------------------------------------------------------------

class NERRunner:
    def __init__(self, config: NERConfig):
        self.config = config
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        if self.config.backend == NERBackend.SPACY:
            import spacy
            self._pipeline = spacy.load(self.config.model_name)
        else:
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=self.config.model_name,
                aggregation_strategy="simple",
                device=self.config.device,
                batch_size=self.config.batch_size,
            )

    def run(self, texts: list[str]) -> list[list[dict]]:
        """Returns one list of entity dicts per input text."""
        self._load()
        if self.config.backend == NERBackend.SPACY:
            return [
                [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    for ent in doc.ents
                ]
                for doc in self._pipeline.pipe(texts)
            ]
        else:
            return [
                [
                    {
                        "text": e["word"],
                        "label": e["entity_group"],
                        "start": e["start"],
                        "end": e["end"],
                        "score": e["score"],
                    }
                    for e in output
                ]
                for output in self._pipeline(texts)
            ]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk(self, text: str) -> list[str]:
        if self.config.strategy == ChunkStrategy.PARAGRAPH:
            return self._by_paragraph(text)
        return self._by_sentence(text)

    def _by_paragraph(self, text: str) -> list[str]:
        return [p.strip() for p in text.split(self.config.paragraph_sep) if p.strip()]

    def _by_sentence(self, text: str) -> list[str]:
        import re
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in raw if s]

        chunks, buf = [], []
        for sent in sentences:
            buf.append(sent)
            if sum(len(s.split()) for s in buf) >= self.config.max_tokens:
                chunks.append(" ".join(buf))
                buf = buf[-self.config.overlap_sentences:] if self.config.overlap_sentences else []

        if buf:
            chunks.append(" ".join(buf))
        return chunks


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_vllm_model(cfg: VLLMModelConfig):
    return ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=cfg.model_type,
        url=cfg.url,
        model_config_dict={
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AgentPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ner = NERRunner(config.ner)
        self.chunker = Chunker(config.chunking)
        self.agents: list[tuple[str, ChatAgent]] = []
        self._build_agents()

    def _build_agents(self):
        for role in self.config.roles:
            agent = ChatAgent(
                system_message=role.system_message,
                model=make_vllm_model(role.model),
            )
            self.agents.append((role.description, agent))

    def run(self, prompt: str, task_id: str = "task-0") -> dict:
        """Run a single prompt through the full agent chain."""
        reasoning_chain = []
        current_input = prompt

        for agent_description, agent in self.agents:
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=current_input,
            )
            response = agent.step(user_msg)
            agent_output = response.msgs[0].content if response.msgs else ""

            reasoning_chain.append({
                "agent": agent_description,
                "input": current_input,
                "reasoning": agent_output,
            })
            current_input = agent_output

        return {
            "id": task_id,
            "original_prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_result": current_input,
        }

    def run_batch(self, prompts: list[str]) -> list[dict]:
        """Run a list of prompts sequentially."""
        return [self.run(p, task_id=f"task-{i}") for i, p in enumerate(prompts)]

    def run_on_document(self, document: str, task_id_prefix: str = "doc") -> dict:
        """Chunk → NER → agent chain per chunk."""
        chunks = self.chunker.chunk(document)
        entities_per_chunk = self.ner.run(chunks)

        chunk_results = []
        for i, (chunk, entities) in enumerate(zip(chunks, entities_per_chunk)):
            entity_summary = (
                ", ".join(f"{e['text']} ({e['label']})" for e in entities) or "none"
            )
            enriched_prompt = f"[Entities: {entity_summary}]\n\n{chunk}"
            result = self.run(enriched_prompt, task_id=f"{task_id_prefix}-chunk-{i}")
            chunk_results.append({
                "chunk_index": i,
                "chunk_text": chunk,
                "entities": entities,
                **result,
            })

        return {
            "document_results": chunk_results,
            "total_chunks": len(chunks),
        }