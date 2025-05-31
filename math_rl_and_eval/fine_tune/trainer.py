import openai
import json

def fine_tune_model(file_id, base_model="gpt-3.5-turbo"):
    job = openai.FineTuningJob.create(
        training_file=file_id,
        model=base_model
    )
    print(f"Fine-tuning started. Job ID: {job['id']}")
    return job['id']

def monitor_job(job_id):
    print("Monitoring job...")
    while True:
        job = openai.FineTuningJob.retrieve(job_id)
        if job["status"] == "succeeded":
            print("✓ Fine-tuning succeeded.")
            print("Model ID:", job["fine_tuned_model"])
            return job["fine_tuned_model"]
        elif job["status"] == "failed":
            raise Exception("✗ Fine-tuning failed.")
