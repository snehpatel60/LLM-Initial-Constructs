# LLM Agent & Evaluation Framework Template

"""
Overview:
- Create a modular LLM agent using OpenAI's GPT (e.g., gpt-4-turbo)
- Add support for tools (calculator, retrieval)
- Design a CLI and prompt interface
- Set up a basic evaluation pipeline (manual + LLM-as-judge)

Dependencies:
- langchain
- requests (for simple web search)
- tiktoken (optional)
- json, logging, datetime, etc.
- rich (for pretty CLI output)
- huggingface_hub (for non-OpenAI models)
- transformers
"""

# ------------------ SETUP ------------------
import os
import json
import logging
from datetime import datetime
from typing import List, Dict

import requests
from rich import print
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Create a global variable for the chat pipeline
chat_pipeline = None

CACHE_DIR = os.path.expanduser("~/.cache/huggingface_models")

def get_chat_pipeline():
    global chat_pipeline
    if chat_pipeline is None:
        model_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return chat_pipeline

# ------------------ AGENT CORE ------------------

class SimpleAgent:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        self.model_name = model_name
        self.pipeline = get_chat_pipeline()

    def call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> str:
        prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        response = self.pipeline(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
        return response.split("Assistant:")[-1].strip()

    def ask(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        reply = self.call_llm(messages)
        return reply

# ------------------ TOOL USE ------------------

class SimpleTools:
    @staticmethod
    def search_web(query: str) -> str:
        try:
            response = requests.get("https://api.duckduckgo.com/", params={
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "skip_disambig": 1
            })
            data = response.json()
            related_topics = data.get("RelatedTopics", [])
            results = [topic["Text"] for topic in related_topics if "Text" in topic]
            return "\n".join(results[:3]) or "No good results."
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def calculate(expr: str) -> str:
        try:
            result = eval(expr)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def help_text() -> str:
        return (
            "[bold blue]Available Commands:[/bold blue]\n"
            "[green]ask: <your prompt>[/green] - Ask the LLM anything.\n"
            "[green]search: <query>[/green] - Perform a DuckDuckGo web search.\n"
            "[green]calc: <expression>[/green] - Evaluate a Python expression (e.g., math).\n"
            "[green]eval: <prompt>|<A>|<B>[/green] - Run an evaluation comparison between two answers.\n"
            "[green]batch_eval: <path_to_json_file>[/green] - Run batch evaluation on a dataset.\n"
            "[green]help[/green] - Show this help message again.\n"
            "[green]exit[/green] - Quit the CLI."
        )

# ------------------ CLI INTERFACE ------------------

def cli_interface():
    agent = SimpleAgent()
    evaluator = Evaluator()
    print("[bold green]Welcome to the LLM Agent CLI! Type 'help' to see available commands.")
    print(SimpleTools.help_text())
    while True:
        user_input = input("\n>>> ")
        if user_input.strip().lower() == "exit":
            break
        elif user_input.strip().lower() == "help":
            print(SimpleTools.help_text())
        elif user_input.startswith("search: "):
            query = user_input.replace("search: ", "")
            print(SimpleTools.search_web(query))
        elif user_input.startswith("calc: "):
            expr = user_input.replace("calc: ", "")
            print(SimpleTools.calculate(expr))
        elif user_input.startswith("ask: "):
            prompt = user_input.replace("ask: ", "")
            print(agent.ask(prompt))
        elif user_input.startswith("eval: "):
            try:
                prompt, a, b = user_input.replace("eval: ", "").split("|", 2)
                print(evaluator.eval_pair(prompt.strip(), a.strip(), b.strip()))
            except ValueError:
                print("[red]Error: Use format 'eval: <prompt>|<A>|<B>'")
        elif user_input.startswith("batch_eval: "):
            path = user_input.replace("batch_eval: ", "").strip()
            try:
                with open(path, 'r') as f:
                    dataset = json.load(f)
                results = evaluator.batch_eval(dataset)
                for i, res in enumerate(results):
                    print(f"Example {i+1}: {res}")
            except Exception as e:
                print(f"[red]Error loading dataset:[/red] {e}")
        else:
            print("[yellow]Unknown command. Type 'help' to see available commands.")

# ------------------ EVALUATION FRAMEWORK ------------------

class Evaluator:
    def __init__(self):
        self.judge = SimpleAgent()

    def eval_pair(self, prompt: str, answer_a: str, answer_b: str) -> str:
        messages = [
            {"role": "system", "content": "You are a judge comparing two assistant answers."},
            {"role": "user", "content": f"Prompt: {prompt}\n\nA: {answer_a}\n\nB: {answer_b}\n\nWhich one is better? Reply with 'A', 'B', or 'Tie'."}
        ]
        decision = self.judge.call_llm(messages)
        return decision if decision else "No decision"

    def batch_eval(self, dataset: List[Dict[str, str]]) -> List[str]:
        results = []
        for item in dataset:
            decision = self.eval_pair(item['prompt'], item['a'], item['b'])
            results.append(decision)
        return results

# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    cli_interface()
