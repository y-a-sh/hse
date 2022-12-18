from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import model
import pandas as pd


app = FastAPI()


@app.on_event("startup")
def initialize():
    model.init_model()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict_df(pd.DataFrame([vars(item)]))[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([vars(item) for item in items])
    return pd.concat([df.drop('selling_price', axis=1), pd.DataFrame(model.predict_df(df), columns=['selling_price'])], axis=1).to_dict(orient="records")


