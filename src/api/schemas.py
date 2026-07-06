from pydantic import BaseModel


class QueryRequest(BaseModel):
    filename: str
    model: str
