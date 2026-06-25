from agent_pipeline import AgentPipeline

# rough shape
class RAGPipeline:
    def index(self, documents: list[str]) -> None:
        # chunk → embed → store in vector DB

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # embed query → similarity search → return top-k chunks

    def retrieve_and_run(self, query: str, agent_pipeline: AgentPipeline) -> dict:
        # retrieve → pass chunks as context → agent_pipeline.run()