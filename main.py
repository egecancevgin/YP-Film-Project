from fastapi import FastAPI
import uvicorn
import os
import tensorflow as tf
from typing import Union
import tensorflow as tf
import tf-keras as keras
from utils import get_model, review_data_pipeline
from models import Review, Response, ListOfReviews
import numpy as np



FASTAPI_HOST = os.getenv('FASTAPI_HOST') if os.getenv('FASTAPI_HOST') else '0.0.0.0'
FASTAPI_PORT = os.getenv('FASTAPI_PORT') if os.getenv('FASTAPI_PORT') else 8000 
RELOAD = False
model = get_model()

app = FastAPI()





@app.get('/')
async def homepage()->dict[str,Union[str,dict]]:
    return {
        'message':'Hello from fastapi',
        'application_status':{
            'host':FASTAPI_HOST,
            'port':FASTAPI_PORT,
            'reload':RELOAD
        }
    }




@app.post('/predict',response_model=Response)
async def calculate_review(data : Union[ListOfReviews,Review]):

    if isinstance(data,Review):
        data = dict(data)
        final_review_embedding = review_data_pipeline(data.get('review'))
        results = model.predict(final_review_embedding)
        results = results[0]
        is_positive = np.argmax(results)
        positivity_score = results[1]
        return {
            "is_positive":is_positive,
            "positivity_score":positivity_score
        }
    else:
        data = dict(data)
        data = data.get('data')
        all_reviews = [v for d in data for k,v in dict(d).items()]
        final_review_embeddings = np.array(
            [review_data_pipeline(review) for review in all_reviews]
        )
        final_review_embeddings = final_review_embeddings.reshape(-1,250)
        results = model.predict(final_review_embeddings)
        is_positive =list(results.argmax(axis=1))
        positivity_scores = [i[1] for i in results]
        return {
            "is_positive":is_positive,
            "positivity_score":positivity_scores
        }




if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=RELOAD
    )