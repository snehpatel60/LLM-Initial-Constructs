import json
from api import get_chat_response
from pathlib import Path
import re

DATA_FILE = Path("data/training_data.jsonl")

def strip_latex(text):
    # Remove \( \), \[ \], and escaped backslashes
    # text = re.sub(r"\\\\", "\\", text)  # Replace double backslashes
    # text = re.sub(r"\\\(", "", text)
    # text = re.sub(r"\\\)", "", text)
    # text = re.sub(r"\\\[", "", text)
    # text = re.sub(r"\\\]", "", text)
    # text = re.sub(r"\\", "", text)
    # text = re.sub(r"\n[", "", text)
    # text = re.sub(r"]\n", "\n", text)
    return text

# 1: "We have the expression a^b * c^d. You may place the numbers 2, 0, 2, and 3 into those variables a, b, c, and d, in any configuration. What is the configuration that yields the maximum value of the expression?"
# 2: "A rectangle, with sides parallel to the x-axis and y-axis, has opposite vertices located at (15, 3) and (16, 5). A line is drawn through points A(0, 0) and B(3, 1). Another line is drawn through points C(0, 10) and D(2, 9). How many points on the rectangle lie on at least one of the two lines?"
# 3: "Carlos took 70% of a whole pie. Maria took one third of the remainder. What portion of the whole pie was left?"
# 4: "A driver travels for 2 hours at 60 miles per hour, during which her car gets 30 miles per gallon of gasoline. She is paid $0.50 per mile, and her only expense is gasoline at $2.00 per gallon. What is her net rate of pay, in dollars per hour, after this expense?"
# 5: "What is the median of the following list of numbers? 1, 2, 3, 4, ... , 2019, 2020, 1^2, 2^2, 3^2, ... , 2019^2, 2020^2"
# 6: "What is the median of the following list of 4040 numbers? 1, 2, 3, 4, ... , 2019, 2020, 1^2, 2^2, 3^2, ... , 2019^2, 2020^2"
# 7: "What is the sum of the digits of the unique positive integer n such that log base 2 (log base 16 of n) = log base 4 (log base 4 of n)"
# 8: "A frog sitting at the point (1, 2) begins a sequence of jumps, where each jump is parallel to one of the coordinate axes and has length 1, and the direction of each jump (up, down, right, or left) is chosen independently at random. The sequence ends when the frog reaches a side of the square with vertices (0,0), (0,4), (4,4), and (4,0). What is the probability that the sequence of jumps ends on one of the two vertical sides of the square?"
# 9: "The a-th root of ( N times the b-th root of (N times the c-th root of N) ) equals the 36-th root of (N to the power of 25). a, b, and c are all integers. What are a, b, and c?"

# From the above, 5 and 6 were very interesting. Wrong answer for 5, even though in the model's response, it said that the list contained 4040 numbers, so it's not like it filled in the missing information incorrectly.
# And it got the correct answer for 6. Wrong answer for 5, correct answer for 6, consistently after multiple queries.
# Wrong answer for 8 over and over, even though it's a very simple recursion problem that I took 2 minutes to solve. Its approach was incorrect? I told it not to simulate the solution lol (cheating)

def evaluate(prompt):
    response = get_chat_response(prompt)
    print("\nResponse:\n", strip_latex(response))
    score = input("Was this correct? (y/n): ").strip().lower()
    if score == 'y':
        save_example(prompt, response)
    if score == 'n':
        # do reinforcement stuff?
        return

def save_example(prompt, response):
    example = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(example) + "\n")
