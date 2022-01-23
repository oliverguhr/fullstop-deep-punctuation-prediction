from model_test_suite import load_json
from tabulate import tabulate

results = load_json("model_final_suite_results_task1.json")


table = []
for model in results["tests"].values():
    model_result = [
        model["id"],
        model["model"],
        ",".join(model["languages"]),
        model["result"][-2]["eval_f1"]
    ]
    table += [model_result]
    #print(model["id"])
    #print(model["result"][-2]["eval_f1"])

table.sort(key=lambda x: x[2]+str(x[3]))
print(tabulate(table, headers=["id","model","lanuages","f1"], tablefmt="github")) # tablefmt="latex"