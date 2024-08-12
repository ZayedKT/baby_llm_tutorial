""" By author:
This is a hand-made Openai agent, adapted from this source: https://til.simonwillison.net/llms/python-react-pattern
Note that neither langchain nor llama_index was used in this code. The mainstream engineers
in the field are not using these tools for their own agents, so I did not use them either.

Note how the agent will turn up its tools iteratively, until it finds a satisfying answer!
"""

import openai
import re
import httpx
import os
from dotenv import load_dotenv

# Load environment variables, where OPENAI_API_KEY is included
load_dotenv("../.env")
 
class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        with openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                temperature=0.0,
                max_tokens=100,
            )
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content

prompt = open("agent_instruction.txt").read()


action_re = re.compile('^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    i = 0
    bot = ChatBot(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(f" -- running Action [{action}] with input [{action_input}]")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


def wikipedia(q):
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]

def calculate(what):
    return eval(what)

known_actions = {
    "wikipedia": wikipedia,
    "calculate": calculate
}

if __name__=="__main__":
    # query("What does England share borders with?")
    while True:
        query_input = input("Enter your question: ")
        print(query(query_input))