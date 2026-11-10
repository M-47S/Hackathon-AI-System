import logging
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from .structs import ChunkCollection
from pathlib import Path


logger = logging.getLogger(__name__)


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.md_parser = MarkdownNodeParser()
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk(self, markdown_text: str, source_name: str = "unknown") -> ChunkCollection:
        collection = ChunkCollection()
        
        if not isinstance(markdown_text, str):
            raise TypeError(
                f"markdown_text должен быть str, получено {type(markdown_text).__name__}. "
                "Прочитайте файл: Path(...).read_text(encoding='utf-8')"
            )
        
        if not markdown_text.strip():
            logger.warning("Получен пустой Markdown документ.")
            return collection

        doc = Document(text=markdown_text, metadata={"source": source_name})
        
        logger.info("Запуск MarkdownNodeParser...")
        parent_nodes = self.md_parser.get_nodes_from_documents([doc])
        
        
        for parent in parent_nodes:
            parent_doc = Document(text = parent.text)            
            child_nodes = self.text_splitter.get_nodes_from_documents([parent_doc])
            
            parent.relationships[NodeRelationship.CHILD] = []
            
            for child in child_nodes:
                child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent.node_id)
                
                parent.relationships[NodeRelationship.CHILD].append(RelatedNodeInfo(node_id=child.node_id))
                
                child.metadata.update(parent.metadata)
                
                collection.add_children(child)
            
            collection.add_parent(parent)
        
        logger.info(f"Создано {len(collection.parents)} родительских и {len(collection.children)} дочерних чанков.")
        
        return collection
            

if __name__ == "__main__":
    doc_path = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\VectorStore\_test_\10-energy\14-tariffs.md")
    chunks_dir = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\VectorStore\chunks")
    chunker = MarkdownChunker()
    
    doc_text = doc_path.read_text(encoding="utf-8")
    
    collection1 = chunker.chunk(doc_text, doc_path.name)
        
    collection1.save_collection(chunks_dir)
    
    collection2 = ChunkCollection.load_collection(chunks_dir)
    
    pid = collection1.children[0].relationships.get(NodeRelationship.PARENT).node_id
    id_ = collection2.parents[0].node_id
    
    print(pid)
    print(id_)
        