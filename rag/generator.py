"""
Generator module - handles LLM inference via HuggingFace Inference API.
Uses free tier of HuggingFace API.
"""

from huggingface_hub import InferenceClient
from typing import List, Tuple


class Generator:
    """Handles text generation using HuggingFace Inference API."""
    
    # Free models available on HF Inference API
    AVAILABLE_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "qwen-2.5": "Qwen/Qwen2.5-72B-Instruct",
        "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
    }
    
    def __init__(self, hf_token: str, model_key: str = "qwen-2.5"):
        """
        Initialize the generator with HuggingFace credentials.
        
        Args:
            hf_token: HuggingFace API token
            model_key: Key from AVAILABLE_MODELS dict
        """
        self.client = InferenceClient(token=hf_token)
        self.model = self.AVAILABLE_MODELS.get(
            model_key, 
            self.AVAILABLE_MODELS["qwen-2.5"]
        )
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Tuple[str, float, dict]],
        max_tokens: int = 512
    ) -> str:
        """
        Generate an answer based on retrieved context.
        
        Args:
            question: The user's question
            context_chunks: List of (text, distance, metadata) from retrieval
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer string
        """
        # Build context string from retrieved chunks
        context_parts = []
        for i, (text, dist, meta) in enumerate(context_chunks, 1):
            source = meta.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        # Construct the prompt
        prompt = f"""Based on the following context, answer the question. 
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{context_str}

Question: {question}

Answer:"""

        try:
            response = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
            )
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_chat(
        self,
        question: str,
        context_chunks: List[Tuple[str, float, dict]],
        max_tokens: int = 512
    ) -> str:
        """
        Generate answer using chat completion format (better for instruction-tuned models).
        
        Args:
            question: The user's question
            context_chunks: List of (text, distance, metadata) from retrieval
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer string
        """
        # Build context string
        context_parts = []
        for i, (text, dist, meta) in enumerate(context_chunks, 1):
            source = meta.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context. 
If the answer cannot be found in the context, say so clearly.
Always be concise and accurate."""
            },
            {
                "role": "user", 
                "content": f"""Context:
{context_str}

Question: {question}"""
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to text generation if chat fails
            return self.generate_answer(question, context_chunks, max_tokens)
