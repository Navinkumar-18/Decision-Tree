from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import io
import uuid

router = APIRouter()

# In-memory storage for demo
datasets = {}
models = {}

class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    preview: List[Dict[str, Any]]

@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    dataset_id = str(uuid.uuid4())
    datasets[dataset_id] = {
        'name': file.filename,
        'df': df
    }
    column_types = {col: str(df[col].dtype) for col in df.columns}
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    preview = df.head(20).to_dict('records')
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        column_types=column_types,
        missing_values=missing,
        preview=preview
    )
