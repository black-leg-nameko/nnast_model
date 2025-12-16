"""
CodeBERT embedding utilities for CPG nodes.

This module provides functions to embed code snippets using CodeBERT,
which is a pre-trained model for code understanding.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Dict, Any
import numpy as np


class CodeBERTEmbedder:
    """CodeBERT embedder for code snippets."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        """
        Initialize CodeBERT embedder.
        
        Args:
            model_name: HuggingFace model name for CodeBERT
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def embed(self, code: str, max_length: int = 512) -> np.ndarray:
        """
        Embed a single code snippet.
        
        Args:
            code: Code snippet string (can be None or empty)
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector (768-dimensional for codebert-base)
        """
        if not code or not code.strip():
            # Return zero vector for empty code
            return np.zeros(768, dtype=np.float32)
        
        # Tokenize
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, codes: List[str], max_length: int = 512, chunk_size: int = 32) -> np.ndarray:
        """
        Embed a batch of code snippets.
        
        Args:
            codes: List of code snippet strings
            max_length: Maximum sequence length
            chunk_size: Process in chunks to avoid OOM (default: 32)
            
        Returns:
            Array of embeddings (batch_size, 768)
        """
        if not codes:
            return np.zeros((0, 768), dtype=np.float32)
        
        # Handle empty codes
        processed_codes = [c if c and c.strip() else "" for c in codes]
        
        # Process in chunks to avoid OOM
        all_embeddings = []
        for i in range(0, len(processed_codes), chunk_size):
            chunk_codes = processed_codes[i:i + chunk_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                chunk_codes,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(chunk_embeddings)
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings.astype(np.float32)
    
    def embed_nodes(self, nodes: List[Dict[str, Any]], max_length: int = 512, chunk_size: int = 32) -> np.ndarray:
        """
        Embed CPG nodes by extracting code from each node.
        
        Args:
            nodes: List of CPG node dictionaries with 'code' field
            max_length: Maximum sequence length
            chunk_size: Process in chunks to avoid OOM (default: 32)
            
        Returns:
            Array of embeddings (num_nodes, 768)
        """
        codes = [node.get("code") or "" for node in nodes]
        return self.embed_batch(codes, max_length=max_length, chunk_size=chunk_size)

