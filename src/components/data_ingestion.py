import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion.")
        try:
            # Read data from the dataset
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Read data from the dataset.")
            
            # creata a directory to save the data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # save the data to a file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info("Splitting the data into train and test sets.")
            # Split the data into train and test sets
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test data to files
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data ingestion completed successfully.")
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()