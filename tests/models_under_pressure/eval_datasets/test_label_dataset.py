import asyncio
from unittest.mock import patch

import pytest

from models_under_pressure.eval_datasets.label_dataset import (
    analyse_stakes,
    label_dataset,
    label_dataset_async,
)
from models_under_pressure.interfaces.dataset import Dataset


@pytest.fixture
def mock_call_llm_async():
    """Fixture to mock the call_llm_async function"""
    with patch(
        "models_under_pressure.eval_datasets.label_dataset.call_llm_async"
    ) as mock:
        # Configure the mock to return different responses based on input
        async def side_effect(messages, model):
            user_message = messages[1]["content"]
            if "high stakes situation" in user_message.lower():
                return {
                    "answer": "high-stakes",
                    "reason": "This is clearly high-stakes",
                }
            elif "low stakes situation" in user_message.lower():
                return {"answer": "low-stakes", "reason": "This is clearly low-stakes"}
            else:
                return {"answer": "ambiguous", "reason": "This situation is ambiguous"}

        mock.side_effect = side_effect
        yield mock


@pytest.fixture
def mock_rubric():
    """Fixture to mock the rubric file"""
    with patch(
        "models_under_pressure.eval_datasets.label_dataset.LABELING_RUBRIC_PATH"
    ) as mock:
        mock.read_text.return_value = "Test rubric content"
        yield mock


@pytest.fixture
def sample_dataset():
    """Fixture to create a sample dataset for testing"""
    return Dataset(
        inputs=[
            "This is a high stakes situation with serious consequences.",
            "This is a low stakes situation with minimal impact.",
            "This is an unclear situation with some implications.",
        ],
        ids=["id1", "id2", "id3"],
        other_fields={"metadata": ["meta1", "meta2", "meta3"]},
    )


class TestAnalyseStakes:
    @pytest.mark.asyncio
    async def test_analyse_stakes(self, mock_call_llm_async, mock_rubric):
        """Test the analyse_stakes function with different inputs"""
        # Test high-stakes situation
        result = await analyse_stakes(
            "This is a high stakes situation", model="test-model"
        )
        assert result["answer"] == "high-stakes"
        assert "high-stakes" in result["reason"]

        # Test low-stakes situation
        result = await analyse_stakes(
            "This is a low stakes situation", model="test-model"
        )
        assert result["answer"] == "low-stakes"
        assert "low-stakes" in result["reason"]

        # Test ambiguous situation
        result = await analyse_stakes(
            "This is an ambiguous situation", model="test-model"
        )
        assert result["answer"] == "ambiguous"


class TestLabelDataset:
    @pytest.mark.asyncio
    async def test_label_dataset_async(self, mock_call_llm_async, sample_dataset):
        """Test the async version of label_dataset"""
        result = await label_dataset_async(
            sample_dataset, model="test-model", max_concurrent=5
        )

        # Check that the result is a LabelledDataset
        assert result.inputs == sample_dataset.inputs
        assert result.ids == sample_dataset.ids

        # Check that labels and explanations were added
        assert "labels" in result.other_fields
        assert "explanation" in result.other_fields

        # Check that all items were labeled
        assert len(result.other_fields["labels"]) == len(sample_dataset.inputs)
        assert len(result.other_fields["explanation"]) == len(sample_dataset.inputs)

        # Check specific labels
        assert result.other_fields["labels"][0] == "high-stakes"
        assert result.other_fields["labels"][1] == "low-stakes"
        assert result.other_fields["labels"][2] == "ambiguous"

        # Check that original fields were preserved
        assert "metadata" in result.other_fields
        assert (
            result.other_fields["metadata"] == sample_dataset.other_fields["metadata"]
        )

        # Verify the mock was called the correct number of times
        assert mock_call_llm_async.call_count == 3

    def test_label_dataset_sync(self, mock_call_llm_async, sample_dataset):
        """Test the synchronous wrapper for label_dataset"""
        with patch(
            "models_under_pressure.eval_datasets.label_dataset.asyncio.run"
        ) as mock_run:
            # Configure mock_run to actually run the coroutine
            mock_run.side_effect = (
                lambda coro: asyncio.get_event_loop().run_until_complete(coro)
            )

            result = label_dataset(sample_dataset, model="test-model", max_concurrent=5)

            # Verify that asyncio.run was called
            mock_run.assert_called_once()

            # Check that the result is a LabelledDataset
            assert result.inputs == sample_dataset.inputs
            assert result.ids == sample_dataset.ids
            assert "labels" in result.other_fields
            assert "explanation" in result.other_fields


@pytest.mark.asyncio
async def test_worker_error_handling(mock_call_llm_async, sample_dataset):
    """Test error handling in the worker function"""

    # Configure the mock to return None for the second item
    async def side_effect(messages, model):
        user_message = messages[1]["content"]
        if "low stakes situation" in user_message.lower():
            return None
        elif "high stakes situation" in user_message.lower():
            return {"answer": "high-stakes", "reason": "This is clearly high-stakes"}
        else:
            return {"answer": "ambiguous", "reason": "This situation is ambiguous"}

    mock_call_llm_async.side_effect = side_effect

    # Test that ValueError is raised when analyse_stakes returns None
    with pytest.raises(ValueError, match="analyse_stakes returned None"):
        await label_dataset_async(sample_dataset, model="test-model", max_concurrent=5)
