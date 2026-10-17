
import os
import sys
from pathlib import Path

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline


# --------------------------------------------------
# Project paths and environment
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="NetworkSecurity ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(
    directory=str(PROJECT_ROOT / "templates")
)


# --------------------------------------------------
# MongoDB connection
# --------------------------------------------------
mongo_db_url = os.getenv("MONGODB_URL_KEY")
client = None
collection = None

if not mongo_db_url:
    logging.warning(
        "MONGODB_URL_KEY is not configured. MongoDB features are unavailable."
    )
else:
    try:
        client = pymongo.MongoClient(
            mongo_db_url,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
        )
        client.admin.command("ping")

        database = client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]

        logging.info("MongoDB connection successful.")

    except Exception:
        logging.exception("MongoDB connection failed.")
        if client is not None:
            client.close()
        client = None
        collection = None


# --------------------------------------------------
# Home and health routes
# --------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/predict", status_code=307)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction page
# --------------------------------------------------
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="table.html",
        context={"table": None, "error": None},
    )


# --------------------------------------------------
# CSV prediction
# --------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict_route(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            return templates.TemplateResponse(
                request=request,
                name="table.html",
                context={
                    "table": None,
                    "error": "Please upload a valid CSV file.",
                },
                status_code=400,
            )

        logging.info("Received prediction file: %s", file.filename)

        df = pd.read_csv(file.file)

        if df.empty:
            return templates.TemplateResponse(
                request=request,
                name="table.html",
                context={
                    "table": None,
                    "error": "The uploaded CSV file is empty.",
                },
                status_code=400,
            )

        model_path = PROJECT_ROOT / "final_model" / "model.pkl"

        if not model_path.is_file():
            logging.error("Model file is missing: %s", model_path)
            return templates.TemplateResponse(
                request=request,
                name="table.html",
                context={
                    "table": None,
                    "error": "The prediction model is unavailable on the server.",
                },
                status_code=500,
            )

        from networksecurity.utils.main_utils.utils import load_object

        model = load_object(str(model_path))
        predictions = model.predict(df)

        if len(predictions) != len(df):
            raise ValueError("Prediction count does not match CSV row count.")

        df["predicted_column"] = predictions

        table_html = df.to_html(
            classes="table table-striped",
            index=False,
            border=0,
            escape=True,
        )

        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={"table": table_html, "error": None},
        )

    except pd.errors.EmptyDataError:
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "table": None,
                "error": "The uploaded file contains no readable CSV data.",
            },
            status_code=400,
        )

    except Exception:
        logging.exception("Prediction failed.")
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "table": None,
                "error": "Prediction failed. Check the server logs for details.",
            },
            status_code=500,
        )

    finally:
        await file.close()


# --------------------------------------------------
# Training route
# --------------------------------------------------
@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()

        return {
            "status": "success",
            "message": "Training pipeline completed.",
        }

    except Exception as e:
        logging.exception("Training pipeline failed.")
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Local execution
# --------------------------------------------------
if __name__ == "__main__":
    app_run(
        app,
        host="0.0.0.0",
        port=8000,
    )
