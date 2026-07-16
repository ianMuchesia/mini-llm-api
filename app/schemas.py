from pydantic import BaseModel,Field





class PredictionRequest(BaseModel):
    prompt:str  = Field(min_length=1)
    max_length:int  = Field(ge=1,le=512,default=100)
    temperature:float = Field(ge=0.1,le=3.0,default=0.8)
    top_k:int = Field(default=50)
    top_p:float = Field(default=0.9)
    
    
class PredictionResponse(BaseModel):
    generated_text:str
    tokens_generated:int
    
    