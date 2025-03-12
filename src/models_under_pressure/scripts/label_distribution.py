from models_under_pressure.config import EVAL_DATASETS_BALANCED, EVAL_DATASETS_RAW
from models_under_pressure.interfaces.dataset import LabelledDataset


def main():
    # Print distributions for raw datasets
    print("\nRAW DATASETS:")
    for name, path in EVAL_DATASETS_RAW.items():
        if path.exists():
            print(f"\n{name.upper()}")
            LabelledDataset.load_from(path).print_label_distribution()

    # Print distributions for balanced datasets
    print("\nBALANCED DATASETS:")
    for name, path in EVAL_DATASETS_BALANCED.items():
        if path.exists():
            print(f"\n{name.upper()}")
            LabelledDataset.load_from(path).print_label_distribution()


if __name__ == "__main__":
    main()
