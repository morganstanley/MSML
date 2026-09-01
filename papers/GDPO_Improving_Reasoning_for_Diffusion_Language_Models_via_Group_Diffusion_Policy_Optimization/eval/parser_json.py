import json
import re
import os
import glob
import tiktoken
from collections import defaultdict
from parser_helper import is_equiv, remove_boxed, last_boxed_only_string


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

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

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                parsed_answer = answer_match.group(1).strip()
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(generated_text)
        total_effective_tokens += effective_tokens

        # Extract available numbers and target from ground_truth
        numbers = []
        target = None

        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            # Fallback to parsing from question if ground_truth is not in expected format
            numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
            if numbers_match:
                numbers_str = numbers_match.group(1)
                numbers = [int(num.strip()) for num in numbers_str.split(",")]

            target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))

        equation = ""
        try:
            equation = remove_boxed(last_boxed_only_string(generated_text))
        except:
            # Try to extract from answer tags
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if answer_match:
                equation = answer_match.group(1).strip()
            else:
                equation = generated_text

        # Replace LaTeX operators with Python operators
        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

        # Check for equation with equals sign and extract only the expression part
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None

        # Validate and evaluate the equation
        is_valid = validate_equation(equation, numbers)
        if is_valid:
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct_cells = total_empty_cells = total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract puzzle
        puzzle_str = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle_str = question[:16]
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        # Extract solution using regex patterns
        solution_str = ""
        patterns = [
            r"<answer>.*?```\s*([\d\s]+)```",
            r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
            r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
            r".*?(\d{16})\s*</answer>",
            r"\b(\d{16})\b",
        ]

        for pattern in patterns:
            if solution_str:
                break
            match = re.search(pattern, raw_generation, re.DOTALL)
            if match and match.group(1).strip():
                solution_str = match.group(1).strip()

        solution_str = re.sub(r"\s", "", solution_str)

        # Handle solution length
        if not solution_str:
            correct_cells = 0
        else:
            if len(solution_str) < 16:
                solution_str = solution_str + "0" * (16 - len(solution_str))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

        accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
            }
        )
    return total_correct_cells, total_empty_cells, processed_items, total_effective_tokens


def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory=".", save_detailed=True):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {"correct": 0, "processed": 0, "accuracy": 0.0, "questions": [], "total_effective_tokens": 0}
    )

    for json_file in json_files:
        if "scratch" in json_file or "llada_math_seq160" in json_file:
            continue
        filename = os.path.basename(json_file)
        setup_name = extract_setup_name(filename)

        if setup_name:
            # print(f"Processing {filename}...")
            if "gsm" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = parse_gsm_answers(
                    json_path=json_file
                )
            elif "math" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = parse_math_answers(
                    json_path=json_file
                )
            elif "countdown" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = parse_countdown_answers(
                    json_path=json_file
                )
            elif "sudoku" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = parse_sudoku_answers(
                    json_path=json_file
                )

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            setups[setup_name]["total_effective_tokens"] += total_effective_tokens
            setups[setup_name]["questions"].extend(detailed_results)

    # Calculate final accuracy and save results
    print("\n===== AGGREGATED RESULTS =====")
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100 if results["processed"] > 0 else 0
        )
        results["avg_effective_tokens"] = results["total_effective_tokens"] / len(results["questions"])
        print(
            f"{setup}: {results['correct']}/{results['processed']} correct ({results['accuracy']:.2f}% accuracy), avg effective tokens: {results['avg_effective_tokens']:.2f}"
        )

        if save_detailed:
            output_filename = f"{setup}_aggregated_results.json"
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            # print(f"Saved detailed results to {output_filename}")


if __name__ == "__main__":
    aggregate_results()
