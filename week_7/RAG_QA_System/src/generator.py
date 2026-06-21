"""
generator.py
============
STEP 7 of the RAG pipeline: Prompt Construction + Generation.

THE CORE IDEA -- "GROUNDING":
This is the step where retrieval results actually become useful. We stuff the
retrieved chunks into the prompt as context, then instruct the model to base
its answer ONLY on that context. This constrains the model's enormous prior
knowledge down to just the document's content -- the entire point of RAG is
defeated if the model answers from its own training data instead of the
retrieved chunks.

WHY THE PROMPT IS WRITTEN THIS SPECIFIC WAY:
- Numbered/labeled context blocks (Source 1, Source 2...) let the model (and
  you, when debugging) trace which chunk a claim came from -- this is what
  makes RAG answers auditable, unlike a black-box fine-tuned model.
- An explicit "if the context doesn't contain the answer, say so" instruction
  is the single most important anti-hallucination lever you have at the
  prompting level. Without it, models default to filling gaps with prior
  knowledge or plausible-sounding fabrication, which silently defeats the
  purpose of grounding.
- Asking for citations of which source number was used is a cheap way to get
  partial faithfulness verification for free -- if the model cites Source 2
  for a claim that's actually only in Source 4, that's an immediate signal
  something's wrong with either retrieval or generation.

TWO SUPPORTED BACKENDS:
- AnthropicGenerator: calls the Claude API. Needs ANTHROPIC_API_KEY set.
- OllamaGenerator: calls a locally running Ollama server (e.g. qwen2.5:0.5b,
  llama3.2) -- free, runs entirely on your machine, no API key needed. Good
  default if you don't want to spend API credits while iterating on the demo.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict

import requests


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """Assemble the retrieved chunks + query into a single grounded prompt."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i} | {chunk.get('source', 'unknown')}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a document question-answering assistant. Answer the question using ONLY the context provided below. Do not use any outside knowledge.

If the context does not contain enough information to answer the question, say "I cannot find this information in the provided documents" -- do not guess or make up an answer.

When you state a fact, mention which Source number it came from (e.g. "(Source 2)").

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    return prompt


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class AnthropicGenerator(BaseGenerator):
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1000):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set the ANTHROPIC_API_KEY environment variable to use this backend.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaGenerator(BaseGenerator):
    """
    Calls a locally running Ollama server. Install Ollama, then run e.g.:
        ollama pull qwen2.5:0.5b
        ollama serve   (usually auto-starts after install)
    """
    def __init__(self, model: str = "qwen2.5:0.5b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


def get_generator(backend: str = "ollama", **kwargs) -> BaseGenerator:
    """Factory function so the pipeline doesn't need to know about backend classes directly."""
    if backend == "anthropic":
        return AnthropicGenerator(**kwargs)
    elif backend == "ollama":
        return OllamaGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
