from dotenv import load_dotenv
from langchain.messages import HumanMessage
load_dotenv()

from graph import graph
from logger import section, log_in, log_out

task = (
    "Write down 150 word brief on the current state of open-source AI Agent frameworks"
)

section("main.py :: graph.invoke START", "bold green")
log_in("task", task)

result = graph.invoke(
    {"messages": [HumanMessage(content=task)]}, {"recursion_limit": 8}
)

section("main.py :: graph.invoke END", "bold green")
log_out("result['messages'] count", len(result["messages"]))

for m in result["messages"]:
    m.pretty_print()