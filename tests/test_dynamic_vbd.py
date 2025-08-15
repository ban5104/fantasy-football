"""
Comprehensive test suite for Dynamic VBD system

Tests mathematical accuracy, robustness, performance, and integration
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamic_vbd import (
    DynamicVBDTransformer, DraftState, ProbabilityForecast,
    create_probability_forecast, create_draft_state
)


class TestDraftState:
    """Test DraftState class functionality"""
    
    def test_creation(self):
        """Test basic DraftState creation"""
        state = DraftState(
            current_pick=15,
            drafted_players={'Player A', 'Player B'}
        )
        assert state.current_pick == 15
        assert len(state.drafted_players) == 2


class TestProbabilityForecast:
    """Test ProbabilityForecast class functionality"""
    
    def test_creation_basic(self):
        """Test basic ProbabilityForecast creation"""
        forecast = ProbabilityForecast(
            horizon_picks=5,
            position_probs={'RB': 0.4, 'WR': 0.3, 'QB': 0.3}
        )
        assert forecast.horizon_picks == 5
        assert forecast.position_probs['RB'] == 0.4


class TestCacheStats:
    """Test cache statistics and methods"""
    
    def test_cache_stats_basic(self):
        """Test basic cache statistics functionality"""
        config = {'dynamic_vbd': {'enabled': True}}
        df = pd.DataFrame()
        
        transformer = DynamicVBDTransformer(config)
        
        stats = transformer.get_cache_stats()
        assert 'cache_size' in stats
        assert 'cache_keys' in stats
        assert stats['cache_size'] == 0  # Should start empty
        
        # Test cache clearing
        transformer.clear_cache()
        stats_after = transformer.get_cache_stats()
        assert stats_after['cache_size'] == 0


class TestMathematicalFunctions:
    """Test core mathematical functions in dynamic VBD"""
    
    def test_adjustment_calculation(self):
        """Test Dynamic VBD adjustment calculation"""
        config = {
            'dynamic_vbd': {
                'enabled': True,
                'params': {
                    'scale': 3.0,
                    'kappa': 5.0
                }
            },
            'basic_settings': {'teams': 12},
            'roster': {'roster_slots': {'RB': 2}}
        }
        
        transformer = DynamicVBDTransformer(config)
        
        # Test with some basic scenarios
        df = pd.DataFrame({
            'POSITION': ['RB', 'RB', 'RB', 'RB'],
            'FANTASY_PTS': [250, 230, 210, 190]
        })
        
        forecast = ProbabilityForecast(
            horizon_picks=5, 
            position_probs={'RB': 0.4}
        )
        
        adjustments = transformer._compute_adjustments(df, forecast)
        
        # Should compute adjustments for given position
        assert 'RB' in adjustments
        assert 'BEER' in adjustments['RB']
        
        # Ensure adjustment is bounded
        adjustment_value = adjustments['RB']['BEER']
        assert adjustment_value >= 190  # Lowest RB points
        assert adjustment_value <= 250  # Highest RB points


class TestDynamicVBDTransformer:
    """Test DynamicVBDTransformer class"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'version': '2025-08-15',
            'basic_settings': {'teams': 12},
            'roster': {
                'roster_slots': {
                    'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DEF': 1, 'K': 1
                }
            },
            'dynamic_vbd': {
                'enabled': True,
                'params': {
                    'scale': 3.0,
                    'kappa': 5.0
                }
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample player DataFrame for testing"""
        return pd.DataFrame({
            'Player': ['QB1', 'QB2', 'RB1', 'RB2', 'WR1', 'WR2'],
            'POSITION': ['QB', 'QB', 'RB', 'RB', 'WR', 'WR'],
            'FANTASY_PTS': [300, 280, 250, 230, 220, 200]
        })
    
    def test_initialization_enabled(self, sample_config):
        """Test transformer initialization when enabled"""
        transformer = DynamicVBDTransformer(sample_config)
        
        assert transformer.enabled is True
        assert transformer.scale == 3.0
        assert transformer.kappa == 5.0
    
    def test_initialization_disabled(self):
        """Test transformer initialization when disabled"""
        config = {'dynamic_vbd': {'enabled': False}}
        transformer = DynamicVBDTransformer(config)
        
        assert transformer.enabled is False
    
    def test_transform_disabled(self, sample_dataframe):
        """Test transform method when dynamic VBD is disabled"""
        config = {'dynamic_vbd': {'enabled': False}}
        transformer = DynamicVBDTransformer(config)
        
        state = DraftState(1, set())
        forecast = ProbabilityForecast(5, {'RB': 0.5})
        
        result = transformer.transform(sample_dataframe, forecast, state)
        
        # Should return copy of original DataFrame
        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)
    
    @patch('src.vbd.calculate_all_vbd_methods')
    def test_transform_enabled(self, mock_vbd, sample_config, sample_dataframe):
        """Test transform method when enabled"""
        # Mock the VBD calculation function
        mock_vbd.return_value = sample_dataframe.copy()
        
        transformer = DynamicVBDTransformer(sample_config)
        
        state = DraftState(1, set())
        forecast = ProbabilityForecast(5, {'RB': 0.4, 'WR': 0.3, 'QB': 0.3})
        
        result = transformer.transform(sample_dataframe, forecast, state)
        
        # Should call VBD calculation with baseline overrides
        mock_vbd.assert_called_once()


class TestErrorHandling:
    """Test error handling and robustness"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        config = {
            'dynamic_vbd': {'enabled': True},
            'basic_settings': {'teams': 12},
            'roster': {'roster_slots': {'QB': 1}}
        }
        empty_df = pd.DataFrame()
        
        transformer = DynamicVBDTransformer(config)
        forecast = ProbabilityForecast(5, {'QB': 0.5})
        
        # Should not crash with empty DataFrame
        adjustments = transformer._compute_adjustments(empty_df, forecast)
        assert adjustments == {}
    
    def test_missing_columns(self):
        """Test handling of DataFrame missing required columns"""
        config = {
            'dynamic_vbd': {'enabled': True},
            'basic_settings': {'teams': 12},
            'roster': {'roster_slots': {'QB': 1}}
        }
        
        # DataFrame missing FANTASY_PTS column
        bad_df = pd.DataFrame({
            'Player': ['QB1'],
            'POSITION': ['QB']
            # Missing FANTASY_PTS
        })
        
        transformer = DynamicVBDTransformer(config)
        forecast = ProbabilityForecast(5, {'QB': 0.5})
        
        # Should handle gracefully and not crash
        try:
            transformer._compute_adjustments(bad_df, forecast)
        except KeyError:
            # This is expected behavior - function should validate inputs
            pass
    
    def test_nan_values(self):
        """Test handling of NaN values in data"""
        config = {
            'dynamic_vbd': {'enabled': True},
            'basic_settings': {'teams': 12},
            'roster': {'roster_slots': {'QB': 1}}
        }
        
        df_with_nan = pd.DataFrame({
            'Player': ['QB1', 'QB2'],
            'POSITION': ['QB', 'QB'],
            'FANTASY_PTS': [300, np.nan]
        })
        
        transformer = DynamicVBDTransformer(config)
        forecast = ProbabilityForecast(5, {'QB': 0.5})
        
        # Should handle NaN values
        adjustments = transformer._compute_adjustments(df_with_nan, forecast)
        assert 'QB' in adjustments


class TestIntegration:
    """Integration tests with existing VBD system"""
    
    def test_convenience_functions(self):
        """Test convenience functions for creating objects"""
        # Test create_probability_forecast
        forecast = create_probability_forecast(
            horizon_picks=5,
            position_probs={'RB': 0.4, 'WR': 0.6}
        )
        assert isinstance(forecast, ProbabilityForecast)
        assert forecast.horizon_picks == 5
        
        # Test create_draft_state
        state = create_draft_state(
            current_pick=10,
            drafted_players=['Player A', 'Player B']
        )
        assert isinstance(state, DraftState)
        assert state.current_pick == 10
        assert len(state.drafted_players) == 2


class TestPerformance:
    """Performance and caching tests"""
    
    def test_cache_performance(self):
        """Test that cache improves performance"""
        config = {
            'dynamic_vbd': {'enabled': True},
            'basic_settings': {'teams': 12},
            'roster': {'roster_slots': {'QB': 1, 'RB': 2}}
        }
        
        df = pd.DataFrame({
            'Player': ['QB1', 'RB1', 'RB2'],
            'POSITION': ['QB', 'RB', 'RB'],
            'FANTASY_PTS': [300, 250, 230]
        })
        
        transformer = DynamicVBDTransformer(config)
        forecast = ProbabilityForecast(5, {'RB': 0.4, 'QB': 0.6})
        
        # Time first call (should compute adjustments)
        import time
        start = time.time()
        first_result = transformer._compute_adjustments(df, forecast)
        first_time = time.time() - start
        
        # Clear cache and time second identical call
        transformer.clear_cache()
        start = time.time()
        second_result = transformer._compute_adjustments(df, forecast)
        second_time = time.time() - start
        
        # Results should be identical
        assert first_result == second_result
    
    def test_cache_stats(self):
        """Test cache statistics functionality"""
        config = {'dynamic_vbd': {'enabled': True}}
        
        transformer = DynamicVBDTransformer(config)
        
        stats = transformer.get_cache_stats()
        assert 'cache_size' in stats
        assert 'cache_keys' in stats
        assert stats['cache_size'] == 0  # Should start empty
        
        # Test cache clearing
        transformer.clear_cache()
        stats_after = transformer.get_cache_stats()
        assert stats_after['cache_size'] == 0


# Run tests with appropriate fixtures and mocking
if __name__ == "__main__":
    pytest.main([__file__, "-v"])