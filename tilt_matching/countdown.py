# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from parsers import Parser, evaluate_equation, validate_equation
from gsm8k import GSM8KDataset
import warnings

CTD_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""



class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        
        # Create prompt matching get_countdown_questions format from data_utils.py
        content = f"{self.system_prompt}\nUsing only the numbers {numbers}, create an arithmetic expression that evaluates to exactly {target}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
        messages = [{"role": "user", "content": content}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            prompt = prompt + "<reasoning>"
        
        return prompt, question, (numbers, target)
