"""Shared pytest fixtures for testing."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_text_data():
    """Provide sample text data for testing."""
    return [
        "This is a simple sentence.",
        "This is a more complex sentence with multiple clauses and various linguistic features.",
        "Short text.",
        "A moderately complex sentence that contains several words and demonstrates average complexity."
    ]


@pytest.fixture
def sample_labels():
    """Provide sample complexity labels."""
    return [0, 1, 0, 1]  # Binary complexity labels


@pytest.fixture
def sample_dataset_dict():
    """Provide a sample dataset dictionary structure."""
    return {
        'train': {
            'text': [
                "Simple sentence one.",
                "Complex sentence with multiple subordinate clauses and advanced vocabulary.",
                "Another simple text.",
                "Moderately complex sentence structure."
            ],
            'label': [0, 1, 0, 1]
        },
        'test': {
            'text': [
                "Test simple sentence.",
                "Test complex sentence with intricate grammatical structures."
            ],
            'label': [0, 1]
        }
    }


@pytest.fixture
def mock_config():
    """Provide mock configuration settings."""
    return {
        'model_name': 'bert-base-uncased',
        'num_epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_length': 128,
        'num_labels': 2,
        'output_dir': './test_output',
        'logging_steps': 100,
        'eval_steps': 500,
        'save_steps': 1000,
        'warmup_steps': 100
    }


@pytest.fixture
def sample_tokenized_data():
    """Provide sample tokenized data structure."""
    return {
        'input_ids': np.array([[101, 2023, 2003, 1037, 6816, 102, 0, 0],
                              [101, 2023, 2003, 2062, 3375, 102, 0, 0]]),
        'attention_mask': np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 0, 0]]),
        'labels': np.array([0, 1])
    }


@pytest.fixture
def mock_model_output():
    """Provide mock model output for testing."""
    return {
        'logits': np.array([[0.8, 0.2], [0.3, 0.7]]),
        'predictions': np.array([0, 1]),
        'probabilities': np.array([[0.8, 0.2], [0.3, 0.7]])
    }


@pytest.fixture
def sample_metrics():
    """Provide sample evaluation metrics."""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85,
        'loss': 0.35
    }


@pytest.fixture
def create_test_file():
    """Factory fixture to create temporary test files."""
    created_files = []
    
    def _create_file(content: str, filename: str = "test_file.txt", temp_dir: Path = None):
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        file_path = temp_dir / filename
        file_path.write_text(content)
        created_files.append(file_path)
        return file_path
    
    yield _create_file
    
    # Cleanup
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()
        if file_path.parent.exists() and not any(file_path.parent.iterdir()):
            file_path.parent.rmdir()


@pytest.fixture
def mock_spacy_doc():
    """Provide mock spaCy document structure."""
    class MockToken:
        def __init__(self, text, pos, lemma, is_alpha=True):
            self.text = text
            self.pos_ = pos
            self.lemma_ = lemma
            self.is_alpha = is_alpha
    
    class MockDoc:
        def __init__(self, tokens):
            self.tokens = [MockToken(**token) for token in tokens]
        
        def __iter__(self):
            return iter(self.tokens)
        
        def __len__(self):
            return len(self.tokens)
    
    return MockDoc([
        {'text': 'This', 'pos': 'DET', 'lemma': 'this'},
        {'text': 'is', 'pos': 'AUX', 'lemma': 'be'},
        {'text': 'a', 'pos': 'DET', 'lemma': 'a'},
        {'text': 'test', 'pos': 'NOUN', 'lemma': 'test'},
        {'text': 'sentence', 'pos': 'NOUN', 'lemma': 'sentence'}
    ])


@pytest.fixture
def mock_transformer_tokenizer():
    """Provide mock transformer tokenizer."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.vocab_size = 30522
        
        def __call__(self, text, max_length=128, padding='max_length', truncation=True, return_tensors=None):
            # Simple mock tokenization
            tokens = text.split()[:max_length-2]  # Account for CLS and SEP
            input_ids = [self.cls_token_id] + list(range(1000, 1000 + len(tokens))) + [self.sep_token_id]
            
            # Pad to max_length
            while len(input_ids) < max_length:
                input_ids.append(self.pad_token_id)
            
            attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]
            
            return {
                'input_ids': input_ids[:max_length],
                'attention_mask': attention_mask[:max_length]
            }
    
    return MockTokenizer()


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and settings."""
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'false')
    monkeypatch.setenv('TRANSFORMERS_VERBOSITY', 'error')