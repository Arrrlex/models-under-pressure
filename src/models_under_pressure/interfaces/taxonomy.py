import json
from pathlib import Path


class Taxonomy:
    """Represents a taxonomy of entities with hierarchy support."""

    def __init__(self, json_path: Path):
        self.json_path = json_path
<<<<<<< HEAD

        # Load data
        self.tree = json.load(open(self.json_path))
        # self.examples = pd.read_csv(self.csv_path)
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
=======
        self.data = json.load(open(self.json_path))
        self.items = list(self.data.keys())


class TopicTaxonomy(Taxonomy):
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.data = json.load(open(self.json_path))
        self.items = self.data["topics"]

>>>>>>> ad2bd97 (Stuff)


class FactorTaxonomy(Taxonomy):
    """Represents a taxonomy of factors with hierarchy support."""

<<<<<<< HEAD
    pass
=======
    def __init__(self, json_path: Path):
        super().__init__(json_path)
>>>>>>> ad2bd97 (Stuff)


if __name__ == "__main__":
    from models_under_pressure.config import (
<<<<<<< HEAD
        CATEGORY_JSON,
        CATEGORY_EXAMPLES_CSV,
        FACTOR_JSON,
        FACTOR_EXAMPLES_CSV,
    )

    category_taxonomy = CategoryTaxonomy(
        csv_path=CATEGORY_EXAMPLES_CSV,
        json_path=CATEGORY_JSON,
=======
        SITUATION_FACTORS_JSON,
        TOPICS_JSON,
>>>>>>> ad2bd97 (Stuff)
    )

    category_taxonomy = TopicTaxonomy(json_path=TOPICS_JSON)

<<<<<<< HEAD
    print("\nCategory taxonomy all nodes:")
    print(category_taxonomy.nodes)

    print("\nFactor taxonomy all nodes:")
    print(factor_taxonomy.nodes)
=======
    factor_taxonomy = FactorTaxonomy(json_path=SITUATION_FACTORS_JSON)
>>>>>>> ad2bd97 (Stuff)
