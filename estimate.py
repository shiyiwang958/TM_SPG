from datasets import load_dataset

# load dataset
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

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

N = 15000
count = 0

for ex in ds.select(range(N)):
    if add_sub_only_possible(ex["nums"], ex["target"]):
        count += 1

prob = count / N

print(f"solvable with only + and -: {count}/{N}")
print(f"estimated probability: {prob:.4f}")