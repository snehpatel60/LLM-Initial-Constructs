# LLM Agent & Evaluation Framework Template

"""
Overview:
- Create a modular LLM agent using OpenAI's GPT (e.g., gpt-4-turbo)
- Add support for tools (calculator, retrieval, formalization)
- Design a CLI and prompt interface
- Set up a basic evaluation pipeline (manual + LLM-as-judge)
- Log evaluation results
- Adapt prompts dynamically based on evaluation feedback

Dependencies:
- openai >= 1.0.0
- python-dotenv
- requests
- tiktoken (optional)
- json, logging, datetime, etc.
- rich
"""

# ------------------ SETUP ------------------
import os
import json
import logging
from datetime import datetime
from typing import List, Dict

import requests
from openai import OpenAI
from dotenv import load_dotenv
from rich import print

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logging
logging.basicConfig(filename="eval_results.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# ------------------ AGENT CORE ------------------

class SimpleAgent:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.formal_prompt_style = "You rewrite informal text into formal English while preserving meaning."

    def call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def ask(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        reply = self.call_llm(messages)
        return reply

    def formalize(self, sentence: str) -> str:
        messages = [
            {"role": "system", "content": self.formal_prompt_style},
            {"role": "user", "content": f"Rewrite this more formally: '{sentence}'"}
        ]
        return self.call_llm(messages)

    def adjust_formal_prompt(self, feedback: str):
        if "formality" in feedback.lower():
            self.formal_prompt_style += " Emphasize high-level vocabulary."
        if "meaning" in feedback.lower():
            self.formal_prompt_style += " Be very careful not to change the original meaning."
        if "grammar" in feedback.lower():
            self.formal_prompt_style += " Ensure perfect grammar."

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
            "[green]formalize: <sentence>[/green] - Make a sentence more formal.\n"
            "[green]search: <query>[/green] - Perform a DuckDuckGo web search.\n"
            "[green]calc: <expression>[/green] - Evaluate a Python expression.\n"
            "[green]eval: <prompt>|<A>|<B>[/green] - Generic evaluation.\n"
            "[green]eval_formalize: <sentence>[/green] - Auto-generate and evaluate two formalizations.\n"
            "[green]batch_eval: <path_to_json_file>[/green] - Batch evaluation.\n"
            "[green]help[/green] - Show this message.\n"
            "[green]exit[/green] - Quit."
        )

# ------------------ EVALUATION FRAMEWORK ------------------

class Evaluator:
    def __init__(self):
        self.judge = SimpleAgent()
        self.generator = SimpleAgent()

    def eval_pair(self, prompt: str, answer_a: str, answer_b: str) -> str:
        messages = [
            {"role": "system", "content": "You are a judge comparing two assistant answers."},
            {"role": "user", "content": f"Prompt: {prompt}\n\nA: {answer_a}\n\nB: {answer_b}\n\nWhich one is better? Reply with 'A', 'B', or 'Tie'."}
        ]
        decision = self.judge.call_llm(messages)
        log_entry = f"PROMPT: {prompt}\nA: {answer_a}\nB: {answer_b}\nDECISION: {decision}"
        logging.info(log_entry)
        return decision if decision else "No decision"

    def eval_formalization(self, sentence: str) -> str:
        prompt_gen = f"Rewrite this more formally: '{sentence}'"
        a = self.generator.call_llm([
            {"role": "system", "content": self.generator.formal_prompt_style},
            {"role": "user", "content": prompt_gen}
        ])
        b = self.generator.call_llm([
            {"role": "system", "content": self.generator.formal_prompt_style},
            {"role": "user", "content": prompt_gen}
        ])
        rubric = (
            "You are evaluating two rewritten versions of a sentence. "
            "Rate based on: (1) increased formality, (2) preservation of meaning, (3) grammatical correctness."
        )
        messages = [
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"Original: {sentence}\n\nA: {a}\n\nB: {b}\n\nWhich is better? Respond with 'A', 'B', or 'Tie'."}
        ]
        feedback = self.judge.call_llm(messages)
        logging.info(f"FORMALIZE EVAL\nOriginal: {sentence}\nA: {a}\nB: {b}\nDecision: {feedback}")
        if "reason" in feedback.lower():
            self.judge.adjust_formal_prompt(feedback)
        return feedback

    def batch_eval(self, dataset: List[Dict[str, str]]) -> List[str]:
        results = []
        for item in dataset:
            decision = self.eval_pair(item['prompt'], item['a'], item['b'])
            results.append(decision)
        return results

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
        elif user_input.startswith("formalize: "):
            sentence = user_input.replace("formalize: ", "")
            print(agent.formalize(sentence))
        elif user_input.startswith("eval: "):
            try:
                prompt, a, b = user_input.replace("eval: ", "").split("|", 2)
                print(evaluator.eval_pair(prompt.strip(), a.strip(), b.strip()))
            except ValueError:
                print("[red]Error: Use format 'eval: <prompt>|<A>|<B>'")
        elif user_input.startswith("eval_formalize: "):
            sentence = user_input.replace("eval_formalize: ", "")
            print(evaluator.eval_formalization(sentence.strip()))
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

# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    cli_interface()
