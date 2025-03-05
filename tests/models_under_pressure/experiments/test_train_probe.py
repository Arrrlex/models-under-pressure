def test_compute_accuracy(
    model_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    layer: int,
):
    print("Loading model...")
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    print("Loading training activations...")
    activations, attention_mask = get_activations(
        model=model,
        config=train_config,
    )
    if any(label == Label.AMBIGUOUS for label in train_config.dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    probe = LinearProbe(_llm=model, layer=train_config.layer)

    print("Training probe...")
    probe.fit(
        X=activations,
        y=train_config.dataset.labels_numpy(),
        attention_mask=attention_mask,
    )

    print("Loading testing activations...")
    activations, attention_mask = get_activations(
        model=model,
        config=test_config,
    )
    if any(label == Label.AMBIGUOUS for label in test_config.dataset.labels):
        raise ValueError("Test dataset contains ambiguous labels")

    print("Computing accuracy...")
    accuracy = compute_accuracy(
        probe,
        test_dataset,
        activations=activations,
        attention_mask=attention_mask,
    )
    print(f"Accuracy: {accuracy}")


def test_activations_on_anthropic_dataset():
    layer = 10
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading dataset...")
    dataset = LabelledDataset.load_from(
        EVAL_DATASETS["anthropic"]["path"],
        field_mapping=EVAL_DATASETS["anthropic"]["field_mapping"],
    )

    train_dataset, test_dataset = create_train_test_split(dataset, split_field="index")

    print("TRAIN:", train_dataset.ids)
    print("TEST:", test_dataset.ids)

    test_get_activations(
        dataset=train_dataset,
        layer=layer,
        model_name=model_name,
    )
    test_compute_accuracy(
        model_name=model_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        layer=layer,
    )
