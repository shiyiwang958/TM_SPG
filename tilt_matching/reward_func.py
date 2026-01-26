# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
import os
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def _debox_if_present(s: str) -> str:
    """
    If s contains a \\boxed{...}, extract its contents; otherwise return s.strip().
    Uses math500_utils helpers already imported at the top of this file.
    """
    if s is None:
        return ""
    s = s.strip()
    try:
        boxed = last_boxed_only_string(s)
        if boxed:
            return remove_boxed(boxed).strip()
    except Exception:
        pass
    return s

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     q = prompts[0][-1]["content"]
#     extracted_responses = [extract_xml_answer(r) for r in responses]

#     RED = "\033[91m"
#     GREEN = "\033[92m"
#     YELLOW = "\033[93m"
#     BLUE = "\033[94m"
#     RESET = "\033[0m"

#     print(
#         "-" * 20,
#         f"\n{RED}Prompt:{RESET}\n{q}\n",
#         "-" * 20,
#         f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
#         "-" * 20,
#         f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
#         "-" * 20,
#         f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
#     )
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    """
    Same reward as the original implementation:
      reward = 2.0 iff extract_xml_answer(response) == answer[i], else 0.0

    Printing (GSM8K):
      - configurable number via kwargs["buffer_print_samples"] (0 disables)
      - prints roughly half reward==2 if possible
      - prints question, reasoning (if present), ground-truth answer, extracted answer, score, full completion
      - rank-0 only if kwargs["rank"] is provided (recommended)
    """
    responses = [completion[0]["content"] for completion in completions]
    # extracted_responses = [_debox_if_present(extract_xml_answer(r)) for r in responses]
    # scores = []
    # for r, a in zip(extracted_responses, answer):
    #     if r == a:
    #         scores.append(2.0)
    #         continue
    #     try:
    #         r_float = float(r)
    #         a_float = float(a)
    #         if r_float == a_float:
    #             scores.append(2.0)
    #             continue
    #     except:
    #         pass
    #     scores.append(0.0)

    extracted_responses = []
    scores = []
    
    for raw_generation, ground_truth in zip(responses, answer):
        parsed_answer = None
        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == float(ground_truth.replace(",", ""))
        print(parsed_answer if parsed_answer is not None else "None", ground_truth, is_correct)
        extracted_responses.append(parsed_answer if parsed_answer is not None else "<no valid answer>")
        if is_correct:
            scores.append(2.0)
        else:
            scores.append(0.0)


    # ---- DEBUG printing: choose ~half from reward==0, remainder from anywhere (no duplicates) ----
    num_print = int(kwargs.get("buffer_print_samples", 0) or 0)
    rank = int(kwargs.get("rank", 0) or 0)  # pass this in from the caller to avoid DDP spam
    if num_print > 0 and rank == 0:
        r = np.asarray(scores, dtype=np.float32)
        max_idxs = np.nonzero(np.isclose(r, 0.0, atol=1e-6, rtol=0.0))[0]
        all_idxs = np.arange(r.shape[0])

        k_total = min(num_print, int(r.shape[0]))
        k_max = min(k_total // 2, int(max_idxs.shape[0]))
        k_any = k_total - k_max

        chosen_parts = []
        if k_max > 0:
            perm = np.random.permutation(max_idxs.shape[0])[:k_max]
            chosen_parts.append(max_idxs[perm])

        if k_any > 0:
            if chosen_parts:
                already = np.zeros(r.shape[0], dtype=bool)
                already[chosen_parts[0]] = True
                pool = all_idxs[~already]
            else:
                pool = all_idxs

            if pool.shape[0] > 0:
                perm = np.random.permutation(pool.shape[0])[: min(k_any, int(pool.shape[0]))]
                chosen_parts.append(pool[perm])

        chosen = np.concatenate(chosen_parts, axis=0) if chosen_parts else np.array([], dtype=np.int64)

        print(
            f"[DEBUG] correctness_reward_func (GSM8K): printing {int(chosen.shape[0])} samples "
            f"(requested={k_total}, reward==2 available={int(max_idxs.shape[0])})",
            flush=True,
        )

        questions = kwargs.get("question", None)
        for idx in chosen.tolist():
            # Question: prefer dataset field; fall back to prompt
            if questions is not None and idx < len(questions):
                q = questions[idx]
            else:
                try:
                    q = prompts[idx][-1]["content"]
                except Exception:
                    q = "<missing>"

            completion_raw = responses[idx]
            completion_pretty = completion_raw.replace("\\n", "\n")

            # Reasoning (optional)
            m = re.findall(r"<reasoning>(.*?)</reasoning>", completion_raw, re.DOTALL)
            reasoning = m[-1].strip() if m else None
            if reasoning is not None:
                reasoning = reasoning.replace("\\n", "\n")

            gt = answer[idx] if idx < len(answer) else None
            extracted = extracted_responses[idx] if idx < len(extracted_responses) else None
            score = float(scores[idx])

            print("--------------------------------", flush=True)
            print(f"Question: {q}", flush=True)
            if reasoning is not None:
                print("\nReasoning:\n", flush=True)
                print(reasoning, flush=True)
            print(f"\nExtracted answer: {extracted}", flush=True)
            print(f"Ground_truth: {gt}", flush=True)
            print(f"Score: {score:.4f}", flush=True)
            print("\nCompletion:\n", flush=True)
            print(completion_pretty, flush=True)

    return scores



def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [_debox_if_present(extract_xml_answer(r)) for r in responses]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.125 if re.fullmatch(r"-?\d+", r) else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [0.25 * count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.0
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    # boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]

    scores = []
    for i, r in enumerate(responses):
        a = remove_boxed(last_boxed_only_string(answer[i]))
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        
        parsed_answer = None
        try:
            parsed_answer = remove_boxed(last_boxed_only_string(r))
        except:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
            if answer_match:
                parsed_answer = answer_match.group(1).strip()
        if not parsed_answer:
            parsed_answer = "<no valid answer>"
        
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, a)
        if is_correct:
            scores.append(2.0)
        else:
            scores.append(0.0)
        print(parsed_answer, a, is_correct)

    # RED = "\033[91m"
    # GREEN = "\033[92m"
    # YELLOW = "\033[93m"
    # BLUE = "\033[94m"
    # RESET = "\033[0m"

    # print(
    #     "-" * 20,
    #     f"\n{RED}Question:{RESET}\n{q}",
    #     "-" * 20,
    #     f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
    #     "-" * 20,
    #     f"\n{BLUE}Response:{RESET}\n{responses[0]}",
    #     "-" * 20,
    #     f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    # )
    # print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    # return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]
    return scores


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.125 for b in boxed_in_answer_rewards]
    return rewards
