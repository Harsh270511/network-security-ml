
import os
import sys
from pathlib import Path

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse
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
# Load environment variables from the project-root .env
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        "MONGODB_URL_KEY is missing from the project-root .env file."
    )
else:
    try:
        client = pymongo.MongoClient(
            mongo_db_url,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
        )

        # Verify the connection rather than assuming it succeeded.
        client.admin.command("ping")

        database = client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]

        logging.info("MongoDB connection successful.")

    except Exception as e:
        logging.exception("MongoDB connection failed: %s", e)
        client = None
        collection = None


# --------------------------------------------------
# Home page
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return RedirectResponse(url="/predict")


# --------------------------------------------------
# Show prediction/upload page
# --------------------------------------------------
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="table.html",
        context={
            "table": None,
            "error": None,
        },
    )


# --------------------------------------------------
# Upload CSV and generate predictions
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
                    "error": "Please upload a valid .csv file.",
                },
                status_code=400,
            )

        logging.info("Received prediction file: %s", file.filename)

        df = pd.read_csv(file.file)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        model_path = PROJECT_ROOT / "final_model" / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        from networksecurity.utils.main_utils.utils import load_object

        model = load_object(str(model_path))
        predictions = model.predict(df)

        df["predicted_column"] = predictions

        table_html = df.to_html(
            classes="table table-striped",
            index=False,
            border=0,
        )

        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "table": table_html,
                "error": None,
            },
        )

    except Exception as e:
        logging.exception("Prediction failed.")

        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "table": None,
                "error": f"Prediction failed: {str(e)}",
            },
            status_code=500,
        )

    finally:
        await file.close()


# --------------------------------------------------
# Start model training
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
# Run application
# --------------------------------------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
