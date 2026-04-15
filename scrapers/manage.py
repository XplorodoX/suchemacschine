import argparse
import os
import subprocess
import sys
from multiprocessing import Process


def run_script(script_path, name):
    print(f"--- Starting {name} ({script_path}) ---")
    try:
        # Resolve path relative to script
        script_abs_path = os.path.join(os.path.dirname(__file__), script_path)
        subprocess.run([sys.executable, script_abs_path], check=True)
        print(f"--- Finished {name} successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error in {name}: {e} ---")

def run_task_group(task_list):
    for script, name in task_list:
        run_script(script, name)

def main():
    parser = argparse.ArgumentParser(description="Manage Scraping and Indexing Tasks")
    parser.add_argument("task", choices=["asta", "starplan", "webpage", "all", "parallel", "list", "hybrid"], help="Task to run")
    
    args = parser.parse_args()
    
    tasks = {
        "asta": [
            ("asta_full_scraper.py", "ASTA Scraping"),
            ("prepare_asta_data.py", "ASTA Data Preparation"),
            ("index_asta_to_qdrant.py", "ASTA Indexing")
        ],
        "starplan": [
            ("starplan_scraper.py", "Starplan Scraping"),
            ("prepare_starplan_data.py", "Starplan Data Preparation"),
            ("index_starplan_to_qdrant.py", "Starplan Indexing")
        ],
        "webpage": [
            ("hs_aalen_extended_scraper.py", "HS Aalen Website Scraping"),
            ("prepare_hs_aalen_extended_data.py", "HS Aalen Data Preparation"),
            ("index_hs_aalen_to_qdrant.py", "HS Aalen Indexing")
        ],
        "hybrid": [
            ("hybrid_indexer.py", "Unified Hybrid Indexing")
        ]
    }
    
    if args.task == "list":
        print("Available tasks: asta, starplan, webpage, all, parallel, hybrid")
        return

    if args.task == "parallel":
        processes = []
        for t in ["asta", "starplan", "webpage"]:
            p = Process(target=run_task_group, args=(tasks[t],))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
            
    elif args.task == "all":
        run_task_group(tasks["asta"] + tasks["starplan"] + tasks["webpage"] + tasks["hybrid"])
        
    elif args.task == "hybrid":
        run_task_group(tasks["hybrid"])
        
    elif args.task in tasks:
        run_task_group(tasks[args.task])
    else:
        print(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()
