from pydantic import BaseModel
from typing import List, Union



class Response(BaseModel):
    is_positive: Union[Union[bool, int], List[Union[bool, int]]]
    positivity_score: Union[float, int, List[float]]


class Review(BaseModel):
    review: str


class ListOfReviews(BaseModel):
    data: List[Review]
