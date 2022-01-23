import json
from model_trainer import ModelTrainer
import sys

def load_json(file):
    with open(file, encoding='utf-8') as json_file:
        return json.load(json_file)    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='train models for SEPP-NLG 2021')
    parser.add_argument("task",help="task 1 or 2")
    args = parser.parse_args()    
    task = args.task

    # read model configs
    suite_config = load_json(f'model_final_suite_task{task}.json')
    completed_runs = load_json(f'model_final_suite_results_task{task}.json')

    # invoke training
    for run_config in suite_config["tests"]:

        run_name = f'{run_config["model"].replace("/","-")}-{"-".join(run_config["languages"])}-{run_config["data_percentage"]}-task{run_config["task"]}'
        if "comment" in run_config:
            run_name +="-"+run_config["comment"]
        
        if str(run_config["id"]) not in completed_runs["tests"]:
            print("invoking test run for model: " +run_name)
        else:
            print("skipping finished run for model: "+run_name)
            continue
        
        try:
            trainer = ModelTrainer(run_name=run_name, **run_config)  
            result = trainer.run_training()
            run_config["result"] = result    
            completed_runs["tests"][run_config["id"]] = run_config
        except:
            print("Unexpected error:", sys.exc_info()[0])         
        

        # write results
        with open(f'model_final_suite_results_task{task}.json', 'w') as outfile:
            json.dump(completed_runs, outfile,indent=4)

