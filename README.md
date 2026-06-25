# CAMEL Workforce 

This is an adapted internal project for public use, functionalities are reduced and limited to provide generalized task-specific uses

A working CAMEL Workforce-style pipelin running entirely on
Ollama/VLLM. Built as a validated architecture to
adopt camel workforce in use for document retrieval analysis and labelization

This pipeline aims to ship with a gold-standard evaluation
harness from day one, instead of asserting zero-shot accuracy without
measurement, allowing for a more streamlined but high insurance output

## Architecture

```
snippet ──> [N family-specialist workers] ──> route_family() picks best hit
        └─> [dimension worker]
        └─> [reviewer worker] ──> QC accept/reject + flag reason
                                        │
                                        v
                          output/workforce_predictions.parquet
                          output/workforce_run_log.json   (full audit trail)
```
## Functionalities 

- Three tier LLM Classifier : Classifier Prompt (Zero-Shot) -> Validation agent (Prompt 1 output as Context Dump) -> Final Review & Confidence (Summarize, Second evaluation layer, Output, and confidence level)

- Scalable Analysis from document level to CSV/Parquet/SQL database

- Chunking using Chonkie Semantic Chunker for a more robust document chunking

- NER Parsing for object driven classifier : detect entities in overall texts

## Long term Objectives 

- Higher level validations : provide more validation metrics based for a more defendable method
- Extraction : outputs should be provided to be more structured either to a data structure / Document interfacer (Pydantic extraction)
- Integrate model orchestrators ( MLFLOW), and parser to Labelstudio/Argilla/Mturk for human validation for given labels (if desired)
