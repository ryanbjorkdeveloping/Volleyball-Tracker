"""
Run volleyball detection on an image or video frame using the Roboflow hosted workflow.
Usage: python scripts/run_inference.py <path_to_image>
"""
import os
import sys
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

required = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_WORKFLOW_ID"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise EnvironmentError(f"Missing env vars: {', '.join(missing)}\nCopy .env.example to .env and fill in your values.")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)


def detect(image_path: str) -> dict:
    result = client.run_workflow(
        workspace_name=os.environ["ROBOFLOW_WORKSPACE"],
        workflow_id=os.environ["ROBOFLOW_WORKFLOW_ID"],
        images={"image": image_path},
        parameters={"classes": "volleyball"},
        use_cache=True,
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect(image_path)
    print(result)
