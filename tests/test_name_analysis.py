#!/usr/bin/env uv python
# /// script
# dependencies = [
#   "pandas",
#   "requests",
#   "textual",
#   "textual-plotext",
#   "python-levenshtein",
#   "jellyfish",
#   "pytest"
# ]
# ///

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import io

# Assuming your main application file is named name_frequency_analysis.py
from name_analysis import DataAnalyzer

# --- Mock Data ---

@pytest.fixture
def mock_data_analyzer():
    """Creates a DataAnalyzer instance with a controlled, in-memory dataset for testing."""
    analyzer = DataAnalyzer(Path("/tmp/fake_data_dir"))
    
    # Create a small, predictable dataset
    mock_data = {
        'Name': ['Mary', 'Anna', 'Emma', 'John', 'William', 'James', 'Mary', 'John', 'Jon', 'Bryan', 'Brian'],
        'Sex': ['F', 'F', 'F', 'M', 'M', 'M', 'F', 'M', 'M', 'M', 'M'],
        'Count': [100, 80, 70, 90, 85, 80, 50, 60, 5, 20, 25],
        'Year': [1990, 1990, 1990, 1990, 1990, 1990, 1991, 1991, 1991, 1990, 1990]
    }
    mock_df = pd.DataFrame(mock_data)
    
    # Manually set the internal DataFrames of the analyzer
    analyzer.df = mock_df
    analyzer.yearly_totals = analyzer.df.groupby("Year")['Count'].sum().reset_index().rename(columns={'Count': 'Total'})
    analyzer.all_time_totals = analyzer.df.groupby(['Name', 'Sex'])['Count'].sum().reset_index()
    analyzer.unique_name_sex_pairs = analyzer.df[['Name', 'Sex']].drop_duplicates().reset_index(drop=True)
    analyzer.first_year_df = analyzer.df.groupby(['Name', 'Sex'])['Year'].min().reset_index().rename(columns={'Year': 'FirstYear'})
    analyzer.phonetic_df = analyzer.unique_name_sex_pairs.copy()
    
    # Mock jellyfish import if it's used in the tested methods
    try:
        import jellyfish
        analyzer.phonetic_df['Metaphone'] = analyzer.phonetic_df['Name'].apply(jellyfish.metaphone)
    except ImportError:
        # If jellyfish isn't installed in the test environment, skip this part
        # A more robust solution might mock the jellyfish library itself.
        pass

    return analyzer

# --- Tests for DataAnalyzer ---

def test_get_name_details(mock_data_analyzer):
    """Test retrieving details for a specific name."""
    details = mock_data_analyzer.get_name_details("Mary", "Female")
    assert details is not None
    assert details['Total Births'] == 150
    assert details['First Appearance'] == 1990
    assert details['Peak Year'] == 1990
    assert details['Peak Year Count'] == 100
    assert details['Most Popular Decade'] == "1990s"

def test_get_top_names(mock_data_analyzer):
    """Test getting the top names for a specific year."""
    top_names = mock_data_analyzer.get_top_names(1990, 2, "Both")
    
    top_females = top_names.get('female')
    top_males = top_names.get('male')

    assert top_females is not None
    assert top_males is not None
    assert len(top_females) == 2
    assert len(top_males) == 2
    assert top_females['Name'].tolist() == ['Mary', 'Anna']
    assert top_males['Name'].tolist() == ['John', 'William']

def test_get_top_names_first_year(mock_data_analyzer):
    """Test that rank movement is 'New' for the first year of data."""
    # Temporarily filter the analyzer's df to simulate 1990 being the first year
    mock_data_analyzer.df = mock_data_analyzer.df[mock_data_analyzer.df.Year == 1990]
    
    top_names = mock_data_analyzer.get_top_names(1990, 1, "Male")
    top_males = top_names.get('male')
    assert top_males is not None
    assert top_males['Mvmt'].iloc[0] == 'New'

def test_get_similar_names_by_spelling(mock_data_analyzer):
    """Test finding names with similar spelling."""
    results, original_count, _ = mock_data_analyzer.get_similar_names("John", "Male", max_distance=1)
    
    assert results is not None
    assert original_count == 150  # 90 (1990) + 60 (1991)
    assert 'Jon' in results['Name'].tolist()
    assert results.iloc[0]['Name'] == 'Jon'
    assert results.iloc[0]['Distance'] == 1

def test_get_phonetically_similar_names(mock_data_analyzer):
    """Test finding names that sound alike using Metaphone."""
    # Ensure jellyfish was available to create the phonetic_df
    if mock_data_analyzer.phonetic_df is None:
        pytest.skip("jellyfish library not available, skipping phonetic test")

    results, _, _ = mock_data_analyzer.get_phonetically_similar_names("Brian", "Male")
    
    assert results is not None
    # Metaphone code for "Brian" and "Bryan" is 'BRN'
    assert 'Bryan' in results['Name'].tolist()
