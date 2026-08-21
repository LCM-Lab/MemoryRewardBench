from typing import Literal, List, Union
from argparse import Namespace
import os, functools, json, numpy as np, pandas as pd

from datasets import Dataset

from evaluating.inference_utils.builder import LLMEvalManagerBuilder


from evaluating.memreward_bench.utils.prompt import (
    extract_judgement
)

from evaluating.deploy_utils.deploy import setup_tokenizer
from evaluating.inference_utils.process import setup_generate_kwargs
from evaluating.memreward_bench.preprocess import (
    MemoryRewardBenchPreprocess
)
from evaluating.base_utils.config import *


def makedirs(func):
    @functools.wraps(func)
    def wrapper(obj, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        return func(obj, path, **kwargs)
    return wrapper



@makedirs
def save_json(obj: Union[list, dict], path: str, **kwargs):
    with open(path,"w") as f:
        json.dump(obj, f, **kwargs) 


@makedirs
def save_jsonl(list_data: list, path: str):
    with open(path, "w") as f:
        for data in list_data:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

def read_jsonl(path: str, encoding: str = 'utf-8') -> List[dict]:
    with open(path, 'r', encoding= encoding) as file:  
        content = [json.loads(line.strip()) for line in file]  
    return content
@makedirs
def save_csv(obj: Union[dict, pd.DataFrame], path: str):
    if isinstance(obj, dict):
        if len(obj) and (not (type(next(iter(obj.values()))) \
                              in (list, tuple))):
            for k,v in obj.items():
                obj[k] = [v]
        obj = pd.DataFrame(obj)[list(obj.keys())]
    obj.to_csv(path)

def get_task_weighted_score(subset):
    tasks = sorted(set(subset['task']))
    scores = 0
    try:
        for t in tasks:
            cset = subset.filter(lambda x: x["task"] == t)
            scores += (sum(cset['result']) / len(cset['result']))
        scores /= len(tasks)
    except ZeroDivisionError: scores = -1
    return scores

class MemoryRewardBenchEvaluator:
    def __init__(self,
                 gpu_list: List[int]= [0, 1, 2, 3, 4, 5, 6, 7],
                 tp_size: int= 1,
                 batch_size: int = 1,
                 backend: Literal["normal_gpu", "vllm_gpu", "vllm_gpu_batch"] = "vllm_gpu",
                 random_sample = None,
                 tasks = [],
                 model_name = "/path/to/model",
                 data_path = "/path/to/data",
                 start_method: Literal["fork", "spawn"] = "fork",
                 generate_kwargs = dict(
                    n = 1,
                    temperature = 0.7,
                    top_p = 0.95,
                    max_tokens = 4096 * 4,
                ),
                max_qsize: int = 2000,
                **kwargs
                ):
        if batch_size != 1: backend = 'vllm_gpu_batch'
        if backend == 'vllm_gpu_batch': assert batch_size != 1

        generate_kwargs = setup_generate_kwargs(
            generate_kwargs, 
            setup_tokenizer(Namespace(model_name = model_name)),
            backend = backend
        )
        self.config = Config(
            gpu_list = gpu_list,
            tp_size = tp_size,
            batch_size = batch_size,
            backend = backend,
            model_name = model_name,
            max_qsize = max_qsize,
            start_method = start_method
        )       

        if "Llama-3" in model_name or \
            "llama3-8b" in model_name and "3.1" not in model_name:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None

        self.benchmark_config = BenchmarkConfig(
                TASKS = tasks,
        )
        self.data_config = DataConfig(
                data_path = data_path,
                model_path = model_name,
                is_api = False,
                random_sample = random_sample
            )
        self.model_config = ModelConfig(
                model_path = model_name,
            )
        
        self.generate_config = GenerateConfig(
                generate_kwargs = dict(
                    **generate_kwargs,
                    **(dict(stop_token_ids = stop_token_ids) \
                       if 'vllm' in backend else dict())
                )
            ) 


    def inference(self, save_path: str, do_evaluate: bool = False):

        MemoryRewardBenchManager = LLMEvalManagerBuilder.build(
            MemoryRewardBenchPreprocess, self.config.backend
        )

        tasks = self.benchmark_config.TASKS
        with MemoryRewardBenchManager(
            tp_size = self.config.tp_size,
            gpu_list = self.config.gpu_list,
            task_info_list = tasks,
            samples_number = sum([len(read_jsonl(f"{self.data_config.data_path}/{task}")) for task in os.listdir(self.data_config.data_path)]),
            data_config = self.data_config,
            model_config = self.model_config,
            generate_config = self.generate_config,
            batch_size = self.config.batch_size,
            max_qsize = self.config.max_qsize,
            start_method = self.config.start_method
        ) as manager:
            for ckpt_results, sub_task in manager.ckpt_result_lists():
                save_jsonl(ckpt_results, f"{save_path}/{sub_task}.jsonl")

        if do_evaluate:
            self.evaluate(save_path, save_path)


    def evaluate(self, results_folder: str, save_path:str = None):
        def compare_shuffled(golden, result):
            if not result: return 0
            return sum(int(a==b) for a, b in zip(golden, result)) / len(golden)
        
        if save_path is None: save_path = results_folder
        # tokenizer = setup_tokenizer(self.config.model_name)

        # get core dataset
        results_grouped = {}
        results_grouped["model"] = self.model_config.model_path
        results_grouped["model_type"] = "Generative RM"
        # results_grouped["chat_template"] = getattr(self.data_config,"chat_template", "zero_shot") \
        # if (not tokenizer.chat_template) else None

        answers = []
        for subset in self.benchmark_config.TASKS:
            answers += read_jsonl(f"{results_folder}/{subset}.jsonl")
        

        result_orders = [extract_judgement(a['output'])\
                                            for a in answers]
        golden_orders = [a['shuffled_order'] for a in answers]

        results = [compare_shuffled(g, r) for g, r in zip(golden_orders, result_orders)]

        task_subsets = [a['subset'] for a in answers]
        lengths = [a['ctx_length'] for a in answers]

        dataset = Dataset.from_dict(dict(
            subset = task_subsets,
            length = lengths,
            result = results,
            task = [a['task'] for a in answers]
        ))

        pairWiseSet = dataset
        pairWise_total_correct = 0
        subset_column_names = []
        subset_results = []
        CTX_LENGTHS = ["8k", "16k", "32k", "64k", "128k"]
        for subset_name in self.benchmark_config.TASKS:
            subset = pairWiseSet.filter(lambda example: example["subset"] == subset_name)
            subtasks = sorted(set(subset["task"]), reverse = "sys" not in subset_name)
            for subtask in subtasks:
                subsubset = subset.filter(lambda example: example["task"] == subtask)
                subsubset_results = []
                cor,tot = 0, 0
                for ctx_length in CTX_LENGTHS:
                    ctx_subsubset = subsubset.filter(lambda example: example["length"] == ctx_length)

                    subsubset_results += [(sum(ctx_subsubset['result']) / len(ctx_subsubset["result"])) if len(ctx_subsubset) else -1]
                    cor += sum(ctx_subsubset['result'])
                    tot += len(ctx_subsubset['result'])
                subset_column_names += [subtask]


                subset_results += [subsubset_results + [cor / tot]] 
            subset_result = []
            cor, tot = 0, 0
            for ctx_length in CTX_LENGTHS:
                ctx_subset = subset.filter(lambda example: example["length"] == ctx_length)

                subset_result += [(sum(ctx_subset['result']) / len(ctx_subset['result'])) if len(ctx_subset['result']) else -1]
                cor += sum(ctx_subset['result'])
                tot += len(ctx_subset['result'])
            subset_column_names += [subset_name]
            
            subset_results += [subset_result + [cor / tot]]


        length_results = []
        cor, tot = 0, 0
        for ctx_length in  CTX_LENGTHS:
            subset = pairWiseSet.filter(lambda example: example["length"] == ctx_length)
            subset_correct = sum(subset["result"])
            length_results += [(subset_correct / len(subset["result"])) if len(subset["result"]) else -1]
            cor += subset_correct
            tot += len(subset['result'])
        subset_column_names += ["Avg."]

        subset_results += [length_results + [cor / tot]]
        
    
        TaskColumn = ["8k", "16k", "32k", "64k", "128k", "AVG."]
        
        subset_results = [[round(k, 5) if type(k) == float else k for k in l] for l in subset_results]
        

        print(len(TaskColumn),f"{save_path}/results.csv")

        save_csv(dict(Task = TaskColumn, **{k: v for k,v in zip(subset_column_names, subset_results)}), f"{save_path}/results.csv")
