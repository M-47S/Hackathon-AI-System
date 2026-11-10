from typing import List, Union, Self
from numpy import ndarray, array
import numpy as np
from dataclasses import dataclass

@dataclass()
class Embedding:
    def __init__(self, vector: Union[List[float], ndarray]):
       self._vector = array(vector, dtype=float).reshape((len(vector), ))

    def __len__(self) -> int:
        return len(self._vector) 
    
    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Embedding):
            ValueError("Vectors must have same types - 'Embedding'")
        return Embedding(self._vector + other.vector)
        
    
    def __mul__(self, other: Self) -> float:
        if not isinstance(other, Embedding):
            ValueError("Vectors must have same types - 'Embedding'")
        return np.sum(self._vector * other.vector)
    
    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self._vector**2)))   # ← евклидова норма
    
    @property
    def vector(self):
        return self._vector
    