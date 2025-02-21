import csv
import json
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd


class Taxonomy:
    """Represents a taxonomy of entities with hierarchy support."""

    def __init__(self, csv_path: Path, json_path: Path):
        self.csv_path = csv_path
        self.json_path = json_path

        # Load data
        self.tree = json.load(open(self.json_path))
        #self.examples = pd.read_csv(self.csv_path)
        # Currently we are using SituationDataInterface to load examples
    
    @property
    def nodes(self) -> List[str]:
        """Returns a list of all node names in the taxonomy, including both leaf and non-leaf nodes."""
        def get_nodes(node: Dict[str, Any]) -> List[str]:
            if not node:
                return []
            
            nodes = list(node.keys())
            for children in node.values():
                nodes.extend(get_nodes(children))
            return nodes

        return get_nodes(self.tree)

    @property
    def leaf_nodes(self) -> List[str]:
        """Returns a list of all leaf node names in the taxonomy.
        
        A leaf node is a node that has an empty dictionary as its value.
        """
        def get_leaves(node: Dict[str, Any]) -> List[str]:
            # If node is an empty dict, the parent key is a leaf
            if not node:
                return []
            
            leaves = []
            for category, children in node.items():
                if children == {}:
                    leaves.append(category)
                else:
                    leaves.extend(get_leaves(children))
            return leaves

        return get_leaves(self.tree)


class CategoryTaxonomy(Taxonomy):
    """Represents a taxonomy of categories with hierarchy support."""
    pass

class FactorTaxonomy(Taxonomy):
    """Represents a taxonomy of factors with hierarchy support."""
    pass


if __name__ == "__main__":
    from models_under_pressure.config import CATEGORY_JSON, CATEGORY_EXAMPLES_CSV, FACTOR_JSON, FACTOR_EXAMPLES_CSV

    category_taxonomy = CategoryTaxonomy(
        csv_path=CATEGORY_EXAMPLES_CSV,
        json_path=CATEGORY_JSON,
    )
    print("Category taxonomy leaf nodes:")
    print(category_taxonomy.leaf_nodes)

    factor_taxonomy = FactorTaxonomy(
        csv_path=FACTOR_EXAMPLES_CSV,
        json_path=FACTOR_JSON,
    )
    print("\nFactor taxonomy leaf nodes:")
    print(factor_taxonomy.leaf_nodes)

    print("\nCategory taxonomy all nodes:")
    print(category_taxonomy.nodes)

    print("\nFactor taxonomy all nodes:")
    print(factor_taxonomy.nodes)