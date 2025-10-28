from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from config.settings import EMBED_DIM

class Document(BaseModel):
    content: str = Field(..., description="Text content chunk")
    embedding: List[float] = Field(..., description="Vector embedding")
    source: str = Field(default="unknown", description="Document source identifier")
    chunk_index: int = Field(default=0, description="Chunk number in document")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # @field_validator("embedding")
    # def validate_embedding_dim(cls, v):
    #     if len(v) != EMBED_DIM:
    #         raise ValueError(f"Embedding dimension must be {EMBED_DIM}, got {len(v)}")
    #     return v