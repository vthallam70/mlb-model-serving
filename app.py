import os,time,joblib
from functools import lru_cache
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
app=FastAPI()
@lru_cache(maxsize=1)
def load_model(path): return joblib.load(path)
class Request(BaseModel): features:list[float]
@app.post('/predict')
def predict(request:Request):
    start=time.perf_counter(); path=os.environ.get('MODEL_PATH','model.joblib')
    try: artifact=load_model(path)
    except FileNotFoundError: raise HTTPException(503,'Model not found; train it first')
    if len(request.features)!=len(artifact['features']): raise HTTPException(422,'Incorrect feature count')
    probability=float(artifact['model'].predict_proba([request.features])[0][1])
    return {'probability':probability,'latency_ms':round((time.perf_counter()-start)*1000,3)}
