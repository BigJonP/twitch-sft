from datasets import load_dataset
import re
import numpy as np

ds = load_dataset("lparkourer10/twitch_chat")

with open("commands.txt", "r") as f:
    commands_list = f.readlines()

commands_list = set([command.strip() for command in commands_list])


def is_command_or_pattern(string):
    pattern = r"(@.*\s|\[.*\].*)"

    first_word = string.split()[0] if string else ""

    if first_word in commands_list or re.search(pattern, string):
        return True
    return False


ds.set_format(type="numpy", columns=["Message"])
ds = ds["train"].filter(lambda x: x["Message"] is not None)
ds = ds.filter(lambda x: not is_command_or_pattern(x["Message"]))
