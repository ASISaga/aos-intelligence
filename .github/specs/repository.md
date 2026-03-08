# aos-intelligence Repository Specification

**Version**: 1.0.0  
**Status**: Active  
**Last Updated**: 2026-03-07

## Overview

`aos-intelligence` is the ML/AI intelligence layer of the **Agent Operating System (AOS)**. It provides ML pipelines, LoRAx multi-adapter serving, DPO training, self-learning systems, knowledge management, and RAG capabilities. It equips `PurposeDrivenAgent` subclasses with domain-adaptive ML capabilities without coupling heavy ML concerns to the lightweight AOS kernel.

## Scope

- Repository role in the AOS ecosystem
- Technology stack and coding patterns
- Testing and validation workflows
- Key design principles for contributors

## Repository Role

| Concern | Owner |
|---------|-------|
| ML pipelines, LoRAx serving, DPO training | **aos-intelligence** (`aos_intelligence.ml`) |
| Knowledge management, RAG engine, interaction learning | **aos-intelligence** (`aos_intelligence.learning`) |
| Evidence retrieval, indexing, precedent engine | **aos-intelligence** (`aos_intelligence.knowledge`) |
| Agent lifecycle, orchestration, messaging, storage | `aos-kernel` |
| Agent base class (`PurposeDrivenAgent`) | `purpose-driven-agent` |

`aos-intelligence` is an **optional add-on** — the package is deliberately decoupled from the AOS kernel. ML dependencies (torch, transformers, trl, peft) are optional extras, guarded with `try/except ImportError`.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Python 3.10+ |
| Core dependency | `pydantic>=2.12.0` |
| ML extras `[ml]` | `transformers`, `torch`, `trl`, `peft`, `accelerate`, `scikit-learn`, `numpy`, `pandas`, `mlflow` |
| RAG extras `[rag]` | `chromadb` |
| Azure Foundry extras `[foundry]` | `azure-ai-projects`, `azure-ai-ml`, `azure-ai-inference`, `azure-identity` |
| Default base model | `meta-llama/Llama-3.1-8B-Instruct` (env default); `meta-llama/Llama-3.3-70B-Instruct` (recommended for production) |
| Tests | `pytest` + `pytest-asyncio` |
| Linter | `pylint` |
| Type checker | `mypy` |
| Formatter | `black`, `isort` |
| Build | `hatchling` |

## Directory Structure

```
aos-intelligence/
├── src/
│   └── aos_intelligence/
│       ├── __init__.py
│       ├── config.py              # MLConfig — single source of truth for all configuration
│       ├── ml/                    # MLPipelineManager, LoRAxServer, DPOTrainer, SelfLearningSystem, FoundryAgentServiceClient
│       ├── learning/              # KnowledgeManager, RAGEngine, InteractionLearner, DomainExpert, LearningPipeline
│       └── knowledge/             # EvidenceRetrieval, IndexingEngine, PrecedentEngine
├── tests/
│   ├── conftest.py
│   ├── test_ml_pipeline.py
│   ├── test_lorax.py
│   ├── test_dpo_trainer.py
│   ├── test_foundry_agent_service.py
│   ├── test_lora_azure.py
│   ├── test_learning.py
│   └── test_knowledge.py
├── examples/                      # Usage examples
├── docs/                          # API reference, guides
├── pyproject.toml                 # Build config, dependencies, pytest/pylint/mypy settings
└── azure.yaml                     # Azure Developer CLI deployment config
```

## Core Patterns

### MLConfig — Single Source of Truth

```python
from aos_intelligence.config import MLConfig

# From environment variables
config = MLConfig.from_env()

# Direct construction
config = MLConfig(
    enable_training=True,
    enable_lorax=True,
    lorax_base_model="meta-llama/Llama-3.1-8B-Instruct",  # default; use Llama-3.3-70B for production
)
```

### Optional ML Dependency Guard

All heavy ML dependencies are optional and must be guarded:

```python
try:
    import torch
    import transformers
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
```

### Storage Backend Injection

Storage backends are dependency-injected — never import storage classes directly:

```python
from aos_intelligence.learning import KnowledgeManager

km = KnowledgeManager(storage_manager=None)   # or pass an AOS storage backend
await km.initialize()
```

### Async IO Pattern

All IO operations are `async`/`await`:

```python
async def main():
    pipeline = MLPipelineManager(config)
    job_id = await pipeline.train_model({...})
    result = await pipeline.infer("adapter-name", "query text")
```

### Logging Convention

Use the module-level logger with `logging.getLogger("AOS.<module>")`:

```python
import logging
logger = logging.getLogger("AOS.ml")
logger.info("Training job started: %s", job_id)
```

## Testing Workflow

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=aos_intelligence --cov-report=term-missing

# Single module
pytest tests/test_ml_pipeline.py -v
pytest tests/test_lorax.py -v
pytest tests/test_learning.py -v
pytest tests/test_knowledge.py -v

# Lint
pylint src/aos_intelligence/

# Type check
mypy src/aos_intelligence/
```

**CI**: GitHub Actions runs `pytest` across Python 3.10, 3.11, and 3.12 on every push/PR to `main`.

→ **CI workflow**: `.github/workflows/ci.yml`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AOS_ENABLE_ML_TRAINING` | `true` | Enable model training |
| `AOS_LORAX_BASE_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | LoRAx base model |
| `AOS_LORAX_PORT` | `8080` | LoRAx server port |
| `AOS_ENABLE_DPO` | `true` | Enable DPO training |
| `AOS_ENABLE_LORAX` | `true` | Enable LoRAx multi-adapter serving |
| `FOUNDRY_ENDPOINT` | `""` | Azure AI Agents endpoint |
| `FOUNDRY_MODEL` | `llama-3.3-70b` | Foundry model name |

## Related Repositories

| Repository | Role |
|-----------|------|
| [purpose-driven-agent](https://github.com/ASISaga/purpose-driven-agent) | PurposeDrivenAgent base class |
| [aos-kernel](https://github.com/ASISaga/aos-kernel) | AOS kernel (storage backends) |
| [AgentOperatingSystem](https://github.com/ASISaga/AgentOperatingSystem) | Full AOS monorepo |
| [leadership-agent](https://github.com/ASISaga/leadership-agent) | LeadershipAgent implementation |

## Key Design Principles

1. **Optional ML deps** — Core package has no heavy ML dependencies; extras are opt-in
2. **Async-first** — All IO is `async`/`await`; use `asyncio_mode = "auto"` in tests
3. **Config-driven** — `MLConfig` is the single source of truth; always construct via `MLConfig(...)` or `MLConfig.from_env()`
4. **Dependency injection** — Storage backends are injected; never imported directly
5. **AOS logger convention** — Use `logging.getLogger("AOS.<module>")` throughout

## References

→ **Agent framework**: `.github/specs/agent-intelligence-framework.md`  
→ **Conventional tools**: `.github/docs/conventional-tools.md`  
→ **Python coding standards**: `.github/instructions/python.instructions.md`  
→ **aos-intelligence instructions**: `.github/instructions/aos-intelligence.instructions.md`
