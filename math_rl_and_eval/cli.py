import typer
from collect.evaluator import evaluate
from fine_tune.formatter import validate_jsonl
from fine_tune.uploader import upload_file
from fine_tune.trainer import fine_tune_model, monitor_job
import json

app = typer.Typer()

@app.command()
def ask(prompt: str):
    prompt = "Present your answer to the following in a readable text format, not in latex or anything similar. " + prompt
    evaluate(prompt)

@app.command()
def finetune():
    validate_jsonl("data/training_data.jsonl")
    file_id = upload_file("data/training_data.jsonl")
    job_id = fine_tune_model(file_id)
    model_id = monitor_job(job_id)
    with open("models/model_registry.json", "w") as f:
        json.dump({"latest_model": model_id}, f, indent=2)

if __name__ == "__main__":
    app()
