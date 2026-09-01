import sys, os
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffu-grpo")))

from utils_code.rewards import code_reward, get_code_format_reward

class TestCodeRewards(unittest.TestCase):
    def setUp(self):
        # Define test cases that will be used by both reward functions
        self.test_cases = {
            "correct": {
                "content": """Let me implement a function to calculate the sum of a list.
```python
def sum_list(numbers):
    return sum(numbers)
```
""",
                "test_cases": [
                    "assert sum_list([1, 2, 3]) == 6",
                    "assert sum_list([]) == 0",
                    "assert sum_list([-1, 1]) == 0",
                    "assert sum_list([5]) == 5"
                ],
                "expected_reward": 1.0,
                "format_reward": 1.0
            },
            "invalid_syntax": {
                "content": """Below is the implementation of the function to calculate the sum of a list.
```python
def sum_list(numbers)
    return sum(numbers)  # Missing colon
```
""",
                "test_cases": [
                    "assert sum_list([1, 2, 3]) == 6",
                    "assert sum_list([]) == 0"
                ],
                "expected_reward": 0.0,
                "format_reward": 0.5
            },
            "incorrect_format": {
                "content": """Let me implement a function to calculate the sum of a list.
```python
def sum_list(numbers):
    return sum(numbers)

""",
                "test_cases": [
                    "assert sum_list([1, 2, 3]) == 6",
                    "assert sum_list([]) == 0"
                ],
                "expected_reward": 0.0,
                "format_reward": 0.0
            },
            "multiple_blocks": {
                "content": """Let me implement a function to calculate the sum of a list.
```python
# First attempt
def sum_list(numbers):
    return 0
```
```python
# Final solution
def sum_list(numbers):
    return sum(numbers)
```
""",
                "test_cases": [
                    "assert sum_list([1, 2, 3]) == 6",
                    "assert sum_list([]) == 0",
                    "assert sum_list([-1, 1]) == 0",
                    "assert sum_list([5]) == 5"
                ],
                "expected_reward": 0.0,
                "format_reward": 0.0
            }
        }

    def test_code_reward(self):
        """Test the code_reward function with various test cases."""
        for case_name, case in self.test_cases.items():
            with self.subTest(case_name=case_name):
                test_completions = [[{"content": case["content"]}]]
                reward_kwargs = {"test_cases": [case["test_cases"]]}
                rewards = code_reward(test_completions, **reward_kwargs)
                self.assertEqual(rewards, [case["expected_reward"]])

    def test_code_format_reward(self):
        """Test the code_format_reward function with various test cases."""
        format_reward = get_code_format_reward(language="python")
        
        for case_name, case in self.test_cases.items():
            with self.subTest(case_name=case_name):
                test_completions = [[{"content": case["content"]}]]
                rewards = format_reward(test_completions)
                self.assertEqual(rewards, [case["format_reward"]])


if __name__ == "__main__":
    unittest.main()
