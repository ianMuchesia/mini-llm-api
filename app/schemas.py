from pydantic import BaseModel





class PredictionRequest(BaseModel):
    prompt:str
    max_length:int
    temperature:float
    top_k:int
    
    
class PredictionResponse(BaseModel):
    generated_text:str
    tokens_generated:int
    
    