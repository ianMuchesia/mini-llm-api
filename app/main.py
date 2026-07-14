import torch
from fastapi import FastAPI
from app.model_loader import load_model,load_tokenizer
from app.schemas import PredictionRequest,PredictionResponse
from fastapi import HTTPException
from fastapi.responses import StreamingResponse







app = FastAPI()

config_filepath = "./experiments/gpu/config.json"

model,vocab = load_model(config_filepath)
tokenizer = load_tokenizer(config_filepath)


@app.get("/")
async def root():
    return {
  "service": "Mini LLM API",
  "status": "running"
}
    
    
@app.get("/health")
async def health():
    return {
  "healthy": True
}
    
    
@app.post("/generate")
async def generate(request: PredictionRequest):
    
    encoded = tokenizer.encode(request.prompt)
    
    if not encoded:
        return HTTPException(status_code=400, detail="Input cannot be empty.")
    
    x = torch.tensor([encoded],dtype=torch.long,device="cpu")
    
    
    
    out = model.generate(x,max_new_chars=request.max_length,temperature=request.temperature,k=request.top_k)
    
    
    final_ids = out.squeeze().cpu().tolist()
    
    
    output = tokenizer.decode(final_ids)
    
    return PredictionResponse(
        generated_text=output,
        tokens_generated=len(output)
    )
    
    
@app.post("/generate_stream")
async def generate_stream(request:PredictionRequest):
    
    encoded = tokenizer.encode(request.prompt)
    
    
    if not encoded:
        raise HTTPException(status_code=400,detail="Input cannot be  empty")
    
    x = torch.tensor([encoded],dtype=torch.long,device="cpu")
    
    
    def token_generate():
        
        for token_tensor in model.stream(x,max_new_chars=request.max_length,
            temperature=request.temperature,
            k=request.top_k
        ):
            
            token_id = token_tensor.squeeze().item()
            
            text_chunk = tokenizer.decode([token_id])
            
            yield text_chunk
            
    return StreamingResponse(token_generate(),media_type="text/event-stream")