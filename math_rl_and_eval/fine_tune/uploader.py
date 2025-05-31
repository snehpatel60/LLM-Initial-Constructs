import openai

def upload_file(filepath):
    resp = openai.File.create(
        file=open(filepath, "rb"),
        purpose="fine-tune"
    )
    return resp["id"]
