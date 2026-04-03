import os
import json
import random
from typing import List
from collections import defaultdict

LEN_TURN = 10
log_file = "./logs/turn_order.json"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

img_data_path = "/mnt/disk2/hachi/data/PairTally/Image"
img_name_list = [f.split('.')[0] for f in os.listdir(img_data_path)]

def one_permutation(nums: List[int]) -> List[int]:
    nums = nums[:]  # copy
    random.shuffle(nums)
    return nums

def make_log():
    """
    Img1: [
            [0,1,2,3,...9],
            [9, permutation()],
            [8, permutation()]
          ]
    """
    img_turn = defaultdict(list)
    for img in img_name_list:
        # per1
        img_turn[img].append(list(range(0, LEN_TURN)))

        # per2
        nums = list(range(0, LEN_TURN-1))
        per = [LEN_TURN-1] + one_permutation(nums)
        img_turn[img].append(per)

        # per3
        nums = list(range(0, LEN_TURN-2)) + [LEN_TURN-1]
        per = [LEN_TURN-2] + one_permutation(nums)
        img_turn[img].append(per)

    # write output
    with open(log_file, 'w') as f:
        json.dump(img_turn, f, indent=2)

if __name__ == "__main__":
    make_log()

    