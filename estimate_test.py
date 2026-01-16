import json

# load dataset from jsonl file
data = []
with open("dataset/countdown_cd3_test.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

def add_sub_only_possible(nums, target):
    k = len(nums)
    # try all +/- assignments
    for mask in range(1 << k):
        s = 0
        for i, n in enumerate(nums):
            sign = -1 if (mask >> i) & 1 else +1
            s += sign * n
        if s == target:
            return True
    return False

N = min(10000, len(data))
count = 0

for ex in data[:N]:
    nums = list(map(int, ex["input"].split(",")))
    target = int(ex["output"])
    if add_sub_only_possible(nums, target):
        count += 1

prob = count / N

print(f"solvable with only + and -: {count}/{N}")
print(f"estimated probability: {prob:.4f}")