from dotenv import load_dotenv
from langchain.messages import HumanMessage

load_dotenv()

from graph import graph
from logger import section, log_in, log_out

task = "Write down 150 word brief on how reranking works."

section("main.py :: graph.invoke START", "bold green")
log_in("task", task)

result = graph.invoke(
    {"messages": [HumanMessage(content=task)], "revisions": 2}, {"recursion_limit": 15}
)

section("main.py :: graph.invoke END", "bold green")
log_out("result['messages'] count", len(result["messages"]))

for m in result["messages"]:
    m.pretty_print()
