"""Unit tests for spatial analysis filtering utilities.

Tests filtering, data loading, and validation logic without running full models.
"""

from __future__ import annotations

import pandas as pd
import pytest

from texas_mushrooms.pipeline.filters import apply_mushroom_filter, filter_by_year
from texas_mushrooms.config.filter_config import MushroomFilter


@pytest.fixture
def sample_mushroom_df() -> pd.DataFrame:
    """Create a sample DataFrame with mushroom observations."""
    return pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=10),
        "species": [
            "Amanita muscaria",
            "Russula mairei",
            "Trametes versicolor",  # Should be filtered (shelf fungus)
            "Lactarius deliciosus",
            "Stereum hirsutum",  # Should be filtered (crust fungus)
            "Agaricus bisporus",
            "Fuligo septica",  # Should be filtered (slime mold)
            "Boletus edulis",
            "Ganoderma lucidum",  # Should be included (whitelisted despite genus)
            "Mycena haematopus",
        ],
        "caption": [
            "A lovely amanita",
            "Pink russula",
            "Turkey tail on a log",  # Contains keyword
            "Lactarius in the woods",
            "Resupinate crust",  # Contains keyword
            "Button mushroom",
            "Dog vomit slime mold",  # Slime mold term
            "Porcini",
            "Reishi - medicinal",  # Whitelisted
            "Bonnet mushroom",
        ],
        "latitude": [30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7, 30.8, 30.9],
        "longitude": [-97.0, -97.1, -97.2, -97.3, -97.4, -97.5, -97.6, -97.7, -97.8, -97.9],
    })


@pytest.fixture
def weather_df() -> pd.DataFrame:
    """Create a sample weather-enriched DataFrame."""
    return pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=365),
        "species": ["Amanita muscaria"] * 365,
        "species_count": [1, 2, 1, 3, 2] * 73,
        "photo_count": [2, 4, 2, 6, 4] * 73,
        "rain_7": [0.5, 1.0, 0.2, 1.5, 0.8] * 73,
        "soil_moisture_mean": [30, 35, 25, 40, 32] * 73,
        "temperature_mean": [15, 16, 14, 18, 17] * 73,
        "relative_humidity_mean": [60, 65, 55, 70, 62] * 73,
        "seasonal_sin": [0.1, 0.2, 0.15, 0.25, 0.12] * 73,
        "seasonal_cos": [0.95, 0.90, 0.93, 0.85, 0.92] * 73,
    })


class TestFilterByYear:
    """Tests for filter_by_year function."""
    
    def test_filter_by_year_basic(self) -> None:
        """Test filtering to 2018-2024 range."""
        df = pd.DataFrame({
            "date": pd.date_range("2015-01-01", periods=10, freq="YE"),
        })
        
        result = filter_by_year(df, date_col="date")
        
        # Should have 2018, 2019, 2020, 2021, 2022, 2023, 2024 = 7 years
        assert len(result) == 7
        assert result["date"].min().year == 2018
        assert result["date"].max().year == 2024
    
    def test_filter_by_year_empty_result(self) -> None:
        """Test filtering when no rows in range."""
        df = pd.DataFrame({
            "date": pd.date_range("2010-01-01", periods=5, freq="YE"),
        })
        
        result = filter_by_year(df, date_col="date")
        
        assert len(result) == 0
    
    def test_filter_by_year_preserves_data(self) -> None:
        """Test that other columns are preserved."""
        df = pd.DataFrame({
            "date": pd.date_range("2018-01-01", periods=5),
            "species": ["Amanita", "Russula", "Lactarius", "Agaricus", "Boletus"],
            "value": [1, 2, 3, 4, 5],
        })
        
        result = filter_by_year(df, date_col="date")
        
        assert len(result) == 5
        assert "species" in result.columns
        assert "value" in result.columns
        assert list(result["species"]) == ["Amanita", "Russula", "Lactarius", "Agaricus", "Boletus"]


class TestApplyMushroomFilter:
    """Tests for apply_mushroom_filter function."""
    
    def test_missing_species_column(self) -> None:
        """Test handling when species column is missing."""
        df = pd.DataFrame({
            "date": ["2020-01-01"],
            "caption": ["A mushroom"],
        })
        
        result = apply_mushroom_filter(df)
        
        # Should return unchanged
        assert len(result) == len(df)
        assert list(result.columns) == ["date", "caption"]
    
    def test_filter_excludes_shelf_fungi(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that shelf fungi (Trametes) are excluded."""
        # Create a filter with just the genera we want to test
        test_filter = MushroomFilter(
            exclude_genera=["Trametes", "Stereum", "Fuligo"],
            exclude_caption_keywords=["turkey tail", "resupinate", "slime mold"],
        )
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Trametes should be filtered out
        assert "Trametes versicolor" not in result["species"].values
        assert len(result) < len(sample_mushroom_df)
    
    def test_filter_excludes_crust_fungi(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that crust fungi (Stereum) are excluded."""
        test_filter = MushroomFilter(exclude_genera=["Stereum"])
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Stereum should be filtered out
        assert "Stereum hirsutum" not in result["species"].values
    
    def test_filter_excludes_slime_molds(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that slime molds (Fuligo) are excluded."""
        test_filter = MushroomFilter(exclude_genera=["Fuligo"])
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Fuligo should be filtered out
        assert "Fuligo septica" not in result["species"].values
    
    def test_filter_includes_whitelisted_species(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that whitelisted species are included despite genus exclusion."""
        test_filter = MushroomFilter(
            exclude_genera=["Ganoderma"],
            include_species=["Ganoderma lucidum"],
        )
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Ganoderma lucidum should be included (whitelisted)
        assert "Ganoderma lucidum" in result["species"].values
    
    def test_filter_excludes_by_caption_keyword(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that captions with excluded keywords are filtered."""
        test_filter = MushroomFilter(exclude_caption_keywords=["turkey tail", "resupinate"])
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Entries with "Turkey tail" caption and "resupinate" should be filtered
        filtered_captions = result["caption"].values
        assert "Turkey tail on a log" not in filtered_captions
        assert "Resupinate crust" not in filtered_captions
    
    def test_filter_count_reduction(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test that filtering reduces the count."""
        test_filter = MushroomFilter(
            exclude_genera=["Trametes", "Stereum", "Fuligo"],
        )
        initial_count = len(sample_mushroom_df)
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=test_filter)
        
        # Should filter out at least Trametes, Stereum, Fuligo
        assert len(result) < initial_count
        assert len(result) >= initial_count - 5


class TestDataIntegration:
    """Integration tests for data loading and preparation."""
    
    def test_weather_data_has_required_columns(self, weather_df: pd.DataFrame) -> None:
        """Test that weather data has all required columns."""
        required_cols = {
            "rain_7",
            "soil_moisture_mean",
            "temperature_mean",
            "relative_humidity_mean",
            "seasonal_sin",
            "seasonal_cos",
            "species_count",
        }
        
        assert required_cols.issubset(set(weather_df.columns))
    
    def test_weather_data_no_nulls_in_key_cols(self, weather_df: pd.DataFrame) -> None:
        """Test that critical weather columns have no nulls."""
        key_cols = [
            "rain_7",
            "temperature_mean",
            "species_count",
        ]
        
        for col in key_cols:
            assert weather_df[col].notna().all(), f"Column {col} has null values"
    
    def test_data_date_parsing(self) -> None:
        """Test that dates are parsed correctly."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        })
        
        df["date"] = pd.to_datetime(df["date"])
        
        assert df["date"].dtype == "datetime64[ns]"
        assert df["date"].min().year == 2020


class TestErrorHandling:
    """Tests for error handling in data loading."""
    
    def test_filter_with_custom_filter_object(self, sample_mushroom_df: pd.DataFrame) -> None:
        """Test applying filter with custom MushroomFilter object."""
        custom_filter = MushroomFilter(exclude_genera=["Trametes"])
        result = apply_mushroom_filter(sample_mushroom_df, mushroom_filter=custom_filter)
        
        # Should still filter correctly with custom object
        assert len(result) < len(sample_mushroom_df)
        assert "Trametes versicolor" not in result["species"].values
    
    def test_filter_by_year_custom_range(self) -> None:
        """Test filtering with custom year range."""
        df = pd.DataFrame({
            "date": pd.date_range("2010-01-01", periods=15, freq="YE"),
        })
        
        result = filter_by_year(df, date_col="date", start_year=2012, end_year=2015)
        
        # Should have 2012, 2013, 2014, 2015 = 4 years
        assert len(result) == 4
        assert result["date"].min().year == 2012
        assert result["date"].max().year == 2015


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
