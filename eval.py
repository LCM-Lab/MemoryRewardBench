
import os, argparse
from evaluating.memreward_bench import MemoryRewardBenchEvaluator



def evaluate(args):

    model_name = os.path.basename(args.model_path.rstrip("/"))

    if "70b" in model_name.lower() or "72b" in model_name.lower(): 
        batch_size = 500
        tp_size = 8
    elif '13b' in model_name.lower() or '14b' in model_name.lower(): 
        batch_size = 5
        tp_size = 2
    else: 
        batch_size = 5
        tp_size = 1
        

    evaluator = MemoryRewardBenchEvaluator(
        model_name = args.model_path,
        data_path = args.data_path,
        backend = 'vllm_gpu_batch',
        batch_size = batch_size,
        tp_size = tp_size,
        generate_kwargs = dict(
            n = 1,
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096 * 4,
        ),
        gpu_list = [0, 1, 2, 3, 4, 5, 6, 7],
        tasks = [
            "Long-context_Reasoning",
            "Multi-turn_Dialogue_Understanding",
            "Long-form_Generation",
        ]
    )


    save_path = f"{os.path.dirname(os.path.abspath(__file__))}/results/{model_name}" \
    if args.save_path is None else args.save_path

    evaluator.inference(save_path = save_path, 
                        do_evaluate = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type = str, required = True
    )
    parser.add_argument(
        "--data-path", type = str, default = "./datas",
        help = "The MemRewardBench Data Path"
    )
    parser.add_argument(
        "--save-path", type = str, default = None,
        help = "The Evaluating Results Saving path"
    )
    
    parser.add_argument(
        "--gpus",type = int, nargs = "+", default = [0, 1, 2, 3, 4, 5, 6, 7]
    )


    args = parser.parse_args()
    evaluate(args)
        