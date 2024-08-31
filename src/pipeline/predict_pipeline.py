import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocess_obj.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                test_preparation_course: str,
                lunch: str,
                writing_score: int,
                reading_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.lunch = lunch
        self.writing_score = writing_score
        self.reading_score = reading_score
    
    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "gender": [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "test preparation course": [self.test_preparation_course],
                "lunch": [self.lunch],
                "writing score": [self.writing_score],
                "reading score": [self.reading_score]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys.exc_info())