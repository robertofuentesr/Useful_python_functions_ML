# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama

# import torch.distributed as dist
# dist.init_process_group(backend='gloo', rank=0, world_size=4, init_method='file:///tmp/sharedfile')

import os 
import torch
#  read my comment here in case you are having problems :)
#  https://github.com/meta-llama/llama/issues/699#issuecomment-1688777017

#os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
#os.environ['NCCL_DEBUG'] = 'INFO'
#torch.distributed.init_process_group(backend="gloo")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """explain this german word in English:
        
        schnurstracks""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    list_text =[]
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        list_text.append(prompt)
        list_text.append(result['generation'])
        list_text.append("\n==================================\n")
    text = ''.join(list_text)
    with open("output.txt", "w") as file:
        file.write(text)

if __name__ == "__main__":
    fire.Fire(main)
