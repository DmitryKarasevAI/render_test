import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from http import HTTPStatus
from typing import Dict, List, Union
from sklearn.linear_model import LinearRegression


app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class Config(BaseModel):
    hyperparameters: Union[Dict, None] = None
    id: str


class FitRequest(BaseModel):
    X: List[List[float]]
    config: Config
    y: List[float]


class ListFitRequest(BaseModel):
    requests: List[FitRequest]


class MessageResponse(BaseModel):
    message: str


class IDRequest(BaseModel):
    id: str


class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[float]


class ModelResponse(BaseModel):
    id: str


class ModelListResponse(BaseModel):
    # Список словарей
    models: Union[List[Dict], None] = None


models = {}

loaded_models = {}


# API endpoints
@app.post("/fit", response_model=List[MessageResponse], status_code=HTTPStatus.CREATED)
async def fit(request: List[FitRequest]):
    # Реализуйте логику обучения и сохранения модели.
    # Обратите внимание на формат входных данных.
    # Обучать линейную регрессию.
    request = request[0].model_dump()
    x = np.array(request['X'])
    y = np.array(request['y'])
    new_model = LinearRegression(**request['config']['hyperparameters'])
    new_model.fit(x, y)
    models.update({request['config']['id']: new_model})
    return [MessageResponse(message=f"Model {request['config']['id']} trained and saved")]


@app.post("/load", response_model=List[MessageResponse])
async def load(request: IDRequest):
    # Реализуйте загрузку обученной модели для инференса.
    request = request.model_dump()
    loaded_models.update({request['id']: models[request['id']]})
    return [MessageResponse(message=f"Model {request['id']} loaded")]


@app.post("/predict", response_model=List[PredictResponse])
async def predict(request: PredictRequest):
    # Реализуйте инференс загруженной модели
    request = request.model_dump()
    ID = request['id']
    X = np.array(request['X'])
    if ID not in loaded_models.keys():
        raise HTTPException(status_code=422, detail="Model not loaded yet")
    else:
        y_pred = loaded_models[ID].predict(X).tolist()
        return [PredictResponse(predictions=y_pred)]


@app.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    # Реализуйте получения списка обученных моделей
    curr_models = []
    for model in models:
        curr_models.append({"id": model})
    return [ModelListResponse(models=curr_models)]


# Реализуйте Delete метод remove_all
@app.delete("/remove_all", response_model=List[MessageResponse])
async def remove_all():
    global models
    responses = []
    for model in models:
        responses.append(MessageResponse(message=f"Model {model} removed"))
    models = {}
    return responses


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
