"""
Mock services for testing.
"""
import pandas as pd

from app.services.data_service import DataService
from app.services.model_service import ModelService


class MockDataService:
    """Mock data service for testing."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url
        self._df = None

    def load_data(self, table_name: str = 'stg_disaster_response'):
        """Mock data loading."""
        # Create a comprehensive mock dataframe with all expected columns
        self._df = pd.DataFrame({
            'id': [1, 2, 3],
            'message': ['Need help with water', 'Offering food supplies', 'Road blocked'],
            'original': ['Need help with water', 'Offering food supplies', 'Road blocked'],
            'genre': ['direct', 'direct', 'news'],
            'related': [1, 1, 1],
            'request': [1, 0, 0],
            'offer': [0, 1, 0],
            'aid_related': [1, 1, 0],
            'medical_help': [0, 0, 0],
            'medical_products': [0, 0, 0],
            'search_and_rescue': [0, 0, 0],
            'security': [0, 0, 0],
            'military': [0, 0, 0],
            'child_alone': [0, 0, 0],
            'water': [1, 0, 0],
            'food': [0, 1, 0],
            'shelter': [0, 0, 0],
            'clothing': [0, 0, 0],
            'money': [0, 0, 0],
            'missing_people': [0, 0, 0],
            'refugees': [0, 0, 0],
            'death': [0, 0, 0],
            'other_aid': [0, 0, 0],
            'infrastructure_related': [0, 0, 1],
            'transport': [0, 0, 1],
            'buildings': [0, 0, 0],
            'electricity': [0, 0, 0],
            'tools': [0, 0, 0],
            'hospitals': [0, 0, 0],
            'shops': [0, 0, 0],
            'aid_centers': [0, 0, 0],
            'other_infrastructure': [0, 0, 0],
            'weather_related': [0, 0, 0],
            'floods': [0, 0, 0],
            'storm': [0, 0, 0],
            'fire': [0, 0, 0],
            'earthquake': [0, 0, 0],
            'cold': [0, 0, 0],
            'other_weather': [0, 0, 0],
            'direct_report': [1, 0, 1]
        })
        return self._df

    def get_data(self):
        """Get the mock data."""
        if self._df is None:
            self.load_data()
        return self._df

    def get_category_columns(self):
        """Get mock category columns."""
        df = self.get_data()
        return df.columns[4:].tolist()


class MockModelService:
    """Mock model service for testing."""

    def __init__(self):
        self._loaded = False

    def load_model(self):
        """Mock model loading."""
        self._loaded = True
        return self

    def predict(self, text: str) -> dict:
        """Mock prediction that returns sample data."""
        # Return a sample prediction with some categories marked as positive
        categories = [
            'related', 'request', 'offer', 'aid_related', 'medical_help',
            'medical_products', 'search_and_rescue', 'security', 'military',
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
            'transport', 'buildings', 'electricity', 'tools', 'hospitals',
            'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
            'direct_report'
        ]

        # Mock some positive predictions based on keywords
        predictions = {}
        probabilities = {}
        text_lower = text.lower()

        for category in categories:
            # Simple keyword-based mock predictions
            if category == 'related':
                # Only mark as related if message contains disaster-related keywords
                disaster_keywords = ['help', 'emergency', 'disaster', 'flood', 'fire', 'earthquake', 'storm', 'medical', 'water', 'food', 'shelter', 'starvation', 'dying', 'child']
                is_related = any(keyword in text_lower for keyword in disaster_keywords)
                predictions[category] = 1 if is_related else 0
                probabilities[category] = 0.9 if is_related else 0.1
            elif category == 'water' and 'water' in text_lower:
                predictions[category] = 1
                probabilities[category] = 0.7
            elif category == 'food' and ('food' in text_lower or 'starvation' in text_lower):
                predictions[category] = 1
                probabilities[category] = 0.8
            elif category == 'aid_related' and ('aid' in text_lower or 'help' in text_lower or 'need' in text_lower):
                predictions[category] = 1
                probabilities[category] = 0.75
            elif category == 'request' and ('need' in text_lower or 'request' in text_lower or 'received nothing' in text_lower):
                predictions[category] = 1
                probabilities[category] = 0.7
            elif category == 'medical_help' and ('medical' in text_lower or 'help' in text_lower or 'dying' in text_lower):
                predictions[category] = 1
                probabilities[category] = 0.6
            elif category == 'direct_report' and ('received' in text_lower or 'nothing' in text_lower):
                predictions[category] = 1
                probabilities[category] = 0.65
            else:
                predictions[category] = 0
                probabilities[category] = 0.1

        # Return in same format as real ModelService for consistency
        return {"labels": predictions, "probabilities": probabilities}
    
    def get_thresholds_map(self) -> dict:
        """Mock thresholds map for testing."""
        # Return default thresholds similar to real ModelService
        return {
            'related': 0.5,
            'request': 0.5,
            'aid_related': 0.5,
            'medical_help': 0.5,
            'food': 0.5,
            'water': 0.5,
            'direct_report': 0.5,
        }
