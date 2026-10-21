# 🛡️ Network Security — End-to-End Machine Learning Project

An end-to-end Machine Learning project designed to analyze network security data, validate datasets, detect data drift, and build a model-training and prediction pipeline.

The project follows a modular ML pipeline architecture, covering data ingestion, validation, transformation, model training, and prediction through a Python-based application.

## 📌 Project Overview

Network security is a critical aspect of protecting systems against malicious activities and suspicious network traffic. This project demonstrates how machine learning workflows can be developed to process network-related datasets and support security-focused classification tasks.

The project focuses on building a structured, maintainable, and reproducible machine learning pipeline using modern Python tools.

### ✨ Key Features

- **Data Ingestion:** Ingest network security data from MongoDB.
- **Data Validation:** Validate datasets and check data quality.
- **Data Transformation:** Prepare data for machine learning.
- **Data Drift Detection:** Generate drift reports to identify changes in data distributions.
- **Model Training:** Train and evaluate machine learning models.
- **Experiment Tracking:** Integrate MLflow for tracking experiments and model metrics.
- **Prediction API:** Use FastAPI to build an interface for generating predictions.
- **Modular Architecture:** Separate pipeline components for easier debugging, maintenance, and extension.

## 🏗️ Project Architecture

```text
                ┌──────────────────────┐
                │   MongoDB Database   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Data Ingestion    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Data Validation   │
                │   + Drift Detection  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Transformation  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Model Training    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation and │
                │      Artifacts       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   FastAPI Prediction │
                │         API          │
                └──────────────────────┘
```

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and numerical computing |
| Scikit-learn | Machine learning and preprocessing |
| MongoDB & PyMongo | Data storage and ingestion |
| FastAPI | API development and prediction interface |
| MLflow | Experiment tracking and ML lifecycle management |
| Matplotlib / Seaborn | Data visualization, where used |
| PyYAML | Configuration and report handling |
| Git & GitHub | Version control |
| Uvicorn | Running the FastAPI application |
| Python `venv` | Virtual environment management |

## 📂 Project Structure

```text
NetworkSecurity/
│
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── entity/
│   ├── configuration/
│   ├── constants/
│   ├── exception/
│   ├── logging/
│   └── utils/
│
├── templates/
│   └── table.html
│
├── app.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
└── README.md
```

*Note: This is a representative structure. Keep the folder names and files aligned with your actual repository.*

## ⚙️ Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone (https://github.com/Harsh270511/network-security-ml)
cd NetworkSecurity
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
MONGODB_URL=your_mongodb_connection_string
```

Replace the placeholder with your MongoDB connection string.

**Important:** Never commit your `.env` file, database credentials, or other secrets to GitHub.

Ensure your code reads the same environment variable name, `MONGODB_URL`, or update the example to match your implementation.

### 5. Run the Training Pipeline

Run the training pipeline using the entry point defined in your project.

For example, if your training pipeline is executable as a Python module:

```bash
python -m networksecurity.pipeline.training_pipeline
```

If your project uses a separate training script or another entry point, use that command instead.

### 6. Run the FastAPI Application

```bash
uvicorn app:app --reload
```

Open the following URL in your browser:

- Application: http://127.0.0.1:8000
- Swagger API documentation: http://127.0.0.1:8000/docs

The application endpoints depend on the routes implemented in `app.py`.

## 🔄 Machine Learning Workflow

### 1. Data Ingestion
- Connect to MongoDB.
- Retrieve the required dataset.
- Export or prepare the data for subsequent pipeline stages.

### 2. Data Validation
- Check the dataset against the configured schema.
- Validate required columns and expected data characteristics.
- Generate validation artifacts and a data drift report.

### 3. Data Transformation
- Apply the required preprocessing steps.
- Transform input features into a format suitable for model training.
- Prepare training and testing datasets.

### 4. Model Training
- Train the configured machine learning model.
- Evaluate its performance using the selected metrics.
- Save the required model and preprocessing artifacts.

### 5. Prediction
- Submit input data through the implemented FastAPI interface.
- Process the input using the corresponding preprocessing pipeline.
- Return predictions from the trained model.

## 📊 Experiment Tracking with MLflow

MLflow can be used to track machine learning experiments, including:

- Model parameters.
- Evaluation metrics.
- Experiment runs.
- Model artifacts, where configured.

To start a local MLflow tracking UI, run:

```bash
mlflow ui
```

Then visit:

http://127.0.0.1:5000

Make sure your training code is configured to log experiments to the tracking location you intend to use.

## 🧪 Testing and Verification

Before considering the pipeline ready for use, verify the following:

- [ ] MongoDB connection works correctly.
- [ ] Data ingestion completes successfully.
- [ ] Data validation passes for the expected dataset.
- [ ] Data drift reports are generated.
- [ ] Data transformation completes without errors.
- [ ] Model training and evaluation complete successfully.
- [ ] Model artifacts are saved and can be loaded.
- [ ] FastAPI starts successfully.
- [ ] The prediction endpoint returns the expected response.
- [ ] Sensitive files and credentials are excluded from version control.

## 🔐 Security Best Practices

- Store connection strings and credentials in environment variables.
- Keep `.env`, private datasets, and sensitive model artifacts out of public repositories.
- Validate uploaded files before processing them.
- Handle exceptions without exposing sensitive information.
- Avoid enabling unrestricted file uploads in a publicly accessible deployment.
- Use appropriate authentication and access controls before exposing prediction endpoints publicly.

## 🚀 Future Improvements

- Add automated unit and integration tests.
- Improve model evaluation and comparison.
- Add model monitoring and prediction logging.
- Improve API authentication and request validation.

These are potential extensions and should not be considered implemented unless they are present in the repository.

## 🎯 Learning Outcomes

This project provides practical experience with:

- Building modular end-to-end ML pipelines.
- Working with MongoDB and structured datasets.
- Performing data validation and drift detection.
- Developing reusable data preprocessing workflows.
- Training and evaluating machine learning models.
- Tracking experiments with MLflow.
- Building prediction APIs with FastAPI.
- Debugging Python applications and managing dependencies.

## 👨‍💻 Author

**Harsh Maurya**

B.Tech — Computer Science and Engineering (Artificial Intelligence)

Interested in Machine Learning, AI Engineering, NLP, and building end-to-end AI applications.

⭐ If you find this project useful, consider giving the repository a star.

*Built as part of my journey toward becoming an AI/ML Engineer.*
