import os
import sys

import pytest


def find_examples(dir=os.path.dirname(os.path.realpath(__file__))):
    scripts = []
    for i in os.listdir(dir):
        i_path = os.path.join(dir, i)
        if os.path.isdir(i_path):
            scripts += find_examples(i_path)
        elif i.endswith(".py") and not i.startswith("_") and not i.startswith("."):
            scripts.append(i_path)
    return scripts


@pytest.mark.parametrize("script", find_examples())
def test_example(script):
    assert os.system(f"{sys.executable} {script}") == 0
