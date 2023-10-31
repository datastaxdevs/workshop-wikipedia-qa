from abc import ABC, abstractmethod

class Embed(ABC):

    @abstractmethod    
    def embed(self, text: str) -> list[float]:
        """generate and return the embedding for text string"""
        pass

    @abstractmethod    
    def embed(self, text: str, instruction: str) -> list[float]:
        """generate and return the embedding for text string 
        with generation instruction. Note only some embedding
        models require an instruction.
        """
        pass