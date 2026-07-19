from dotenv import load_dotenv
load_dotenv()

from graph import graph

task = (
    "Write down 150 word brief on the current state of open-source AI Agent frameworks"
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": task}, {"recursion_limit": 8}]}
)


for m in result["messages"]:
    m.pretty_print()