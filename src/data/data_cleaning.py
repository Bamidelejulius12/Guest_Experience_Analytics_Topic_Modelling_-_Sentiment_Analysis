from src.data.data_ingestion import data_loader
import re
from src.logger import configure_logger
from src.exception import MyException

import sys

logging = configure_logger()

class DataCleaning:
    
    def clean_review(self, text) -> str:
        # Convert to string
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove emojis
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        # Keep letters, numbers and spaces
        text = re.sub(r'[^a-z0-9\s]',' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+',' ',text).strip()

        return text

    def clean_review_column(self, dataframe):
        """apply text cleaning using our clean review function on the customer review text"""
        try:
            customer_review_data = dataframe.copy()
            customer_review_data["cleaned_review_text"] = (customer_review_data["review_text"].map(self.clean_review))
            logging.info(customer_review_data.head())

            return customer_review_data
        except Exception as e:
            logging.error(f"error occurred during data cleaning {e}")
            raise MyException(e, sys)

def Data_cleaner():
    customer_data = data_loader()
    DataCleaningEngine = DataCleaning()
    cleaned_customer_review_data = DataCleaningEngine.clean_review_column(customer_data)

    return cleaned_customer_review_data

# Data_cleaner()

        