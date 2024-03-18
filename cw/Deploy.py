import pickle
import uvicorn
import pandas as pd
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# Inicia API
app = FastAPI()

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.get('/predict')
def predict(merchant_id: int, user_id : int, 
            transaction_amount: int,device_id:int):
    df_input = pd.DataFrame([dict(merchant_id:=merchant_id, 
                                 user_id=user_id,transaction_amount=transaction_amount,
                                 device_id=device_id)])
    output = model.predict(df_input)[0]
    return output

# Cria página inicial
@app.get('/')
def home():
    return 'Welcome to Chargeback prediction!'


# Executa API
if __name__ == '__main__':
    uvicorn.run(app)

