from unittest.mock import MagicMock, patch

import pytest

from models_under_pressure.eval_datasets.anthropic_dataset import (
    load_anthropic_dataset,
    parse_messages,
)


@pytest.fixture
def mock_load_dataset():
    with patch(
        "models_under_pressure.eval_datasets.anthropic_dataset.load_dataset"
    ) as mock:
        # Create mock dataset
        mock_sample1 = {
            "chosen": "Human: Question 1\n\nAssistant: Answer 1",
            "rejected": "Human: Question 1\n\nAssistant: Bad answer 1",
        }
        mock_sample2 = {
            "chosen": "Human: Question 2\n\nAssistant: Answer 2",
            "rejected": "Human: Question 2\n\nAssistant: Bad answer 2",
        }

        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [mock_sample1, mock_sample2]
        mock.return_value = mock_dataset

        yield mock


class TestParseMessages:
    def test_basic_conversation(self):
        text = "Human: Hello\n\nAssistant: Hi there"
        messages = parse_messages(text)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there"

    def test_multiline_messages(self):
        text = (
            "Human: Hello\n\nHow are you?\n\nAssistant: I'm good\n\nThanks for asking"
        )
        messages = parse_messages(text)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello\n\nHow are you?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "I'm good\n\nThanks for asking"

    def test_system_message(self):
        text = "You are a helpful assistant.\n\nHuman: Hello\n\nAssistant: Hi there"
        messages = parse_messages(text)

        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi there"

    def test_empty_input(self):
        text = ""
        messages = parse_messages(text)
        assert len(messages) == 0

    def test_only_human_message(self):
        text = "Human: Hello there"
        messages = parse_messages(text)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello there"

    def test_only_assistant_message(self):
        text = "Assistant: Hello there"
        messages = parse_messages(text)

        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hello there"


class TestLoadAnthropicDataset:
    @patch("models_under_pressure.eval_datasets.anthropic_dataset.load_dataset")
    def test_load_anthropic_dataset(self, mock_load_dataset):
        # Create mock dataset
        mock_sample1 = {
            "chosen": "Human: Question 1\n\nAssistant: Answer 1",
            "rejected": "Human: Question 1\n\nAssistant: Bad answer 1",
        }
        mock_sample2 = {
            "chosen": "Human: Question 2\n\nAssistant: Answer 2",
            "rejected": "Human: Question 2\n\nAssistant: Bad answer 2",
        }

        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [mock_sample1, mock_sample2]
        mock_load_dataset.return_value = mock_dataset

        # Call the function
        dataset = load_anthropic_dataset(split="test")

        # Verify load_dataset was called correctly
        mock_load_dataset.assert_called_once_with("Anthropic/hh-rlhf", split="test")

        # Check the returned dataset
        assert len(dataset.inputs) == 4  # 2 samples × 2 (chosen/rejected)
        assert len(dataset.ids) == 4

        # Check IDs format
        assert dataset.ids[0] == "test_0_chosen"
        assert dataset.ids[1] == "test_0_rejected"
        assert dataset.ids[2] == "test_1_chosen"
        assert dataset.ids[3] == "test_1_rejected"

        # Check other fields
        assert dataset.other_fields["split"] == ["test", "test", "test", "test"]
        assert dataset.other_fields["category"] == [
            "chosen",
            "rejected",
            "chosen",
            "rejected",
        ]
        assert dataset.other_fields["index"] == [0, 0, 1, 1]

        # Check message content
        assert len(dataset.inputs[0]) == 2  # Human and Assistant messages
        assert dataset.inputs[0][0].role == "user"
        assert dataset.inputs[0][0].content == "Question 1"
        assert dataset.inputs[0][1].role == "assistant"
        assert dataset.inputs[0][1].content == "Answer 1"

        assert len(dataset.inputs[1]) == 2
        assert dataset.inputs[1][0].role == "user"
        assert dataset.inputs[1][0].content == "Question 1"
        assert dataset.inputs[1][1].role == "assistant"
        assert dataset.inputs[1][1].content == "Bad answer 1"
