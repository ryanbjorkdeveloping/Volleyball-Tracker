import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

required = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}\nCopy .env.example to .env and fill in your values.")

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
version = project.version(int(os.environ["ROBOFLOW_VERSION"]))

dataset = version.download("yolov8", location="data/raw")
print(f"Dataset downloaded to: {dataset.location}")
print(f"data.yaml path: {dataset.location}/data.yaml")
