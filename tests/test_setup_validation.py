"""Validation tests to verify testing infrastructure setup."""

import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Test suite to validate testing infrastructure setup."""
    
    def test_pytest_working(self):
        """Test that pytest is working correctly."""
        assert True
    
    def test_python_version(self):
        """Test that Python version meets requirements."""
        assert sys.version_info >= (3, 8)
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check key files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "transformer").exists()
        
        # Check test structure
        tests_dir = project_root / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()
        assert (tests_dir / "conftest.py").exists()
        assert (tests_dir / "unit").exists()
        assert (tests_dir / "integration").exists()
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True
    
    def test_fixtures_available(self, temp_dir, sample_text_data, mock_config):
        """Test that shared fixtures are available."""
        assert temp_dir.exists()
        assert isinstance(sample_text_data, list)
        assert len(sample_text_data) > 0
        assert isinstance(mock_config, dict)
        assert 'model_name' in mock_config
    
    def test_mock_tokenizer_fixture(self, mock_transformer_tokenizer):
        """Test that mock tokenizer fixture works."""
        result = mock_transformer_tokenizer("test sentence")
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert len(result['input_ids']) == len(result['attention_mask'])
    
    def test_sample_dataset_fixture(self, sample_dataset_dict):
        """Test that sample dataset fixture works."""
        assert 'train' in sample_dataset_dict
        assert 'test' in sample_dataset_dict
        assert 'text' in sample_dataset_dict['train']
        assert 'label' in sample_dataset_dict['train']
    
    def test_create_test_file_fixture(self, create_test_file, temp_dir):
        """Test the create_test_file fixture."""
        content = "This is test content."
        test_file = create_test_file(content, "test.txt", temp_dir)
        
        assert test_file.exists()
        assert test_file.read_text() == content


class TestCoverageConfiguration:
    """Test coverage configuration and reporting."""
    
    def test_coverage_can_track(self):
        """Simple test to ensure coverage tracking works."""
        def covered_function():
            return "This function should be covered"
        
        result = covered_function()
        assert result == "This function should be covered"
    
    def test_uncovered_branch(self):
        """Test with conditional logic for coverage."""
        def conditional_function(condition):
            if condition:
                return "condition true"
            else:
                return "condition false"
        
        # Test both branches
        assert conditional_function(True) == "condition true"
        assert conditional_function(False) == "condition false"


def test_module_level_function():
    """Test that module-level test functions work."""
    assert 1 + 1 == 2


@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parametrized(input_value, expected):
    """Test parametrized test functionality."""
    assert input_value * 2 == expected