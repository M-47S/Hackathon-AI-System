from llama_index.core.schema import TextNode, NodeRelationship
from typing import List, Dict, Union, Literal
from pathlib import Path
import json

class ChunkCollection:
    """
    Контейнер для хранения чанков.
    Разделяет родительские и дочерние чанки для удобства.
    """
    def __init__(self):
        self._parents: List[TextNode] = []
        self._children: List[TextNode] = []
        
        #Cловарь для мгновенного поиска родителя по ID: {parent_id: TextNode}
        self._parent_map: Dict[str, TextNode] = {}
        
        self.pname = Path("parents.json")
        self.cname = Path("children.json")

    def __len__(self) -> int:
        return len(self._children)

    def __save_file(self, ListNode: List[TextNode], fpath: Path):
        data = [node.to_dict() for node in ListNode]
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def __load_file(self, fpath: Path) -> List[TextNode]:
        if not fpath.exists():
            raise FileNotFoundError(f"Файл не найден: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [TextNode.from_dict(item) for item in raw]
    
    def add_children(self, nodes: Union[List[TextNode], TextNode]) -> None:
        if isinstance(nodes, list):
            self._children.extend(nodes)
        else:
            self._children.append(nodes)
            
    def add_parent(self, node: TextNode) -> None:
        self._parents.append(node)
        
        self._parent_map[node.node_id] = node
        
    def get_parent_by_child(self, child_node: TextNode) ->  TextNode | None:
        """Возвращает родительский чанк для заданного дочернего."""
        
        parent_rel = child_node.relationships.get(NodeRelationship.PARENT)
        
        if parent_rel:
            result = self._parent_map.get(parent_rel.node_id)
        else:
            result = None
            
        return result
    
    def save_collection(self, path_to_dir: Path | str):
        if isinstance(path_to_dir, str):
            path_to_dir: Path = Path(path_to_dir)
            
        ppath = path_to_dir / self.pname
        ppath.touch()
        self.__save_file(self._parents, ppath)
        
        cpath = path_to_dir / self.cname
        cpath.touch()
        self.__save_file(self._children, cpath)

    @classmethod
    def load_collection(cls, path_to_dir: Path | str) -> "ChunkCollection":
        """Загружает коллекцию из папки с parents.json и children.json."""
        path_to_dir = Path(path_to_dir)
        collection = cls()

        for parent in collection.__load_file(path_to_dir / collection.pname):
            collection.add_parent(parent)

        collection.add_children(collection.__load_file(path_to_dir / collection.cname))
        return collection
    
    @property
    def children(self):
        return self._children
    
    @property
    def parents(self):
        return self._parents