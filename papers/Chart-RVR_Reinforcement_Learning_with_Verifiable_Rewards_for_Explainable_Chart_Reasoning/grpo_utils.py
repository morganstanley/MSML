import torch
from datasets import load_dataset
import re
import json

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

import matplotlib.pyplot as plt


text_reward_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda().eval()

def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$" # blocks=2
        # pattern = r"^<type>.*?</type>\s*<think>.*?</think>\s*<answer>.*?</answer>" # blocks=3
        # pattern = r"^<type>.*?</type>\s*<table>.*?</table>\s*<think>.*?</think>\s*<answer>.*?</answer>" # blocks=4
        

        # pattern = r"^<think>.*?<type>*?</type>*?<table>*?</table>.*?</think>\n<answer>.*?</answer>$" # blocks=2
        
        # formatted
        # pattern = r"^<think>\n<type>.*?</type>\n<table>.*?</table>.*?</think>\n<answer>.*?</answer>$"


        # pattern = r"^<think>\n<type>.*?</type>\n.*?</think>\n<answer>.*?</answer>$"

        # pattern = r"^<think>\n.*?</think>\n<answer>.*?</answer>$"
        # pattern = r"^<think>.*?</think><answer>.*?</answer>$"

        pattern = r"^<think>\n<type>.*?</type>\n<table>.*?</table>.*?</think>\n<answer>.*?</answer>$"


        # TAG_RE = re.compile(r"<think>\s*[\s\S]+?\s*</think>\s*<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)

        # TAG_RE = re.compile(r"<answer>\s*[\s\S]+?\s*</answer>\s*\Z",flags=re.DOTALL | re.IGNORECASE,)
        
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        # matches = [TAG_RE.search(content) for content in completions]

        # rewards = [3.0 if match else 0.0 for match in matches]
        rewards = [2.0 if match else 0.0 for match in matches]

        print(completions)
        print("Format rewards:", rewards)
        return rewards


def compare_tables(pred, gt):
    # print("#################")
    # print(pred)
    # print("#################")
    # print(gt)
    # print("#################")

    try:
        gt['columns'] = sorted(gt['columns'], key=lambda x: x.lower())
        pred['columns'] = sorted(pred['columns'], key=lambda x: x.lower())
        

    except Exception as e:
        print(e)
        pass

    try:
        gt['rows'] = sorted([g for g in gt['rows']])
        pred['rows'] = sorted([g for g in pred['rows']])
    except Exception as e:
        print(e)
        pass

    reward = 0.0
    for col in range(len(pred["columns"])):
        if pred["columns"][col].lower() == gt["columns"][col].lower():
            reward += 0.5*float(1/len(pred["columns"]))

    for row in range(len(pred["rows"])):
        for row_id in range(len(pred["rows"][row])):
            if pred["rows"][row][row_id] == gt["rows"][row][row_id]:
                reward += 0.5*float(1.0/len(pred["rows"]))
    return reward

def table_style_reward(completions, table, **kwargs):
    rewards = []
    if table is None:
        print("No table provided for table style reward.")
        return [0.0] * len(completions)
    
    for completion,tab in zip(completions, table):
        reward = 0.0

        tab_struct = completion.split("<table>")[-1].strip().split("</table>")[0].strip()

        if "```json" in tab_struct:
                block = tab_struct.split("```json")[-1].split("```")[0].strip()
        else:
            block = tab_struct.replace('\n', '').strip("\n").strip()

        # print(block)
        try:
            jj = json.loads(block, parse_int=str, parse_float=str, parse_constant=str)
            reward += 0.5            # parseable JSON
        except:
            print("Failed to parse JSON")
        
        try:
            reward += compare_tables(jj, tab) # custom function to compare tables
        except Exception as e:
            print(f"Failed to compare tables: {e}")

   # not parseable JSON
        try:
            if set(jj) == {"columns", "rows"}:
                reward += 0.25            # wrong keys
        except:
            print("Set compare fail")
        rewards.append(reward)
    print("Table style rewards:", rewards)
    return rewards

def num_token_reward(completions, **kwargs):
    _PATTERNS = [
            re.compile(r"<type>"),
            re.compile(r"</type>"),
            re.compile(r"<think>"),
            re.compile(r"</think>"),
            re.compile(r"<answer>"),
            re.compile(r"</answer>"),
            re.compile(r"<table>"),
            re.compile(r"</table>"),
            re.compile(r"<think>\n<type>"), # Order is important
            re.compile(r"</type>\n<table>"), # Order is important
            ]
    rewards = []
    for completion in completions:
        reward = 0.0
        # reward = int(all(len(p.findall(completion)) == 1 for p in _PATTERNS))
        reward = 2*int(all(len(p.findall(completion)) == 1 for p in _PATTERNS))

        rewards.append(reward)


    print("Count rewards:", rewards)
    return rewards


def accuracy_reward(completions, label: list[str], **kwargs):
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    # print(completions)

    for completion, sol in zip(completions, label):
        # print(completion)
        # print(label)
        try:
            # print("Completion: ", completion)
            if "<answer>" not in completion:
                gold_parsed = []
            elif "<think>" not in completion:
                gold_parsed = []
            else:
                gold_parsed = completion.split("<answer>")[-1].strip().split("</answer>")[0].strip()
                gold_parsed = gold_parsed.rstrip('.') if gold_parsed.endswith('.') else gold_parsed
            # print("GOLD PARSED:", gold_parsed)
        except Exception:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                sol = float(sol)+1e-6  # to avoid zero division errors
                gold_parsed = float(gold_parsed)
                reward = int(float(abs(gold_parsed - sol)/sol) <= 0.05)
                reward = float(reward)
            except Exception as e:
                # print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                # print(sol, gold_parsed)
                try:
                    reward = int(sol.lower() == gold_parsed.lower())
                    reward = float(reward)
                except:
                    reward = 0.0
        else:
            # fallback to text match
           reward = 0.0

        rewards.append(reward)
    print("Rewards Accuracy:", rewards)
    return rewards


# Ensure that the length of the reasoning is sufficient (50 tokens and 3 steps minimum)
def length_think_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        rationale = completion.split("<think>")[-1].strip().split("</think>")[0].strip()
        if len(rationale) > 150:
            reward +=  1.0
        if len(rationale) > 250:
            reward -=  1.0

        if len(rationale) > 70:
            reward +=  1.0
        if len(rationale) > 150:
            reward -=  1.0

        # text_rationale = completion.split("</table>")[-1].strip().split("<answer>")[0].strip()
        # if len(text_rationale) > 100:
        #     reward +=  1.0
        # if len(text_rationale) > 800:
        #     reward -=  1.0

        # steps = text_rationale.split(".")
        steps = rationale.split("<step-")

        # if len(steps) >= 3:
        reward += min(0.25*len(steps), 1.5)
        
        rewards.append(reward)

        if len(rationale) > 500:
            reward = 0
    print("Length Rewards:", rewards)
    return rewards


def chart_type_reward(completions, chart_type, **kwargs):
    rewards = []
    for completion, c_type in zip(completions, chart_type):
        reward = 0.0
        try:
            pred_type = completion.split("<type>")[-1].strip().split("</type>")[0].strip().lower()
            if pred_type == c_type.lower():
                reward = 1.0
        except Exception as e:
            reward = 0.0
        rewards.append(reward)
    print("Graph Type Rewards:", rewards)
    return rewards


## graph specific rewards here


def process_style_reward(completions, reasoning, **kwargs):
    rewards  = []
    for completion, reason in zip(completions, reasoning):
        steps = completion.split("</table>")[-1].strip().split("</think>")[0].strip()
        pred_steps = steps.split("<step-")
        gt_steps = reason.split("<step-")
        reward = 0.0

        # if len(pred_steps) == len(gt_steps):
        #     reward += 0.5
        #     for st, gst in zip(pred_steps[:len(gt_steps)], gt_steps):
        #         if text_sim(st, gst)> 0.8:
        #             reward += 0.5
        # else: 
        #     reward += text_sim(steps, reason)

        reward += text_sim(steps, reason)
        
        rewards.append(reward)

        # print("#"*20)
        # print(steps)
        # print(reasoning[0])
        # print("#"*20)
    print("Process Style Rewards:", rewards)
    return rewards

def text_sim(pr, gt):
    p_emb = text_reward_model.encode(pr, convert_to_tensor=True, device='cuda')
    gt_emb = text_reward_model.encode(gt, convert_to_tensor=True, device='cuda')
    cos = F.cosine_similarity(p_emb, gt_emb, dim=-1)
    return cos.max(dim=0).values.mean().item()


# def chart_consistency_reward(completions, prompts, chart_type, **kwargs):
#     rewards = []
#     for completion, pr, c_type in zip(completions,prompts, chart_type):
#         # 1. extract table + answer
#         try:
#             # print("Completion: ", completion)
#             if "<answer>" not in completion:
#                 gold_parsed = []
#             elif "<think>" not in completion:
#                 gold_parsed = []
#             else:
#                 gold_parsed = completion.split("<answer>")[-1].strip().split("</answer>")[0].strip()
#                 gold_parsed = gold_parsed.rstrip('.') if gold_parsed.endswith('.') else gold_parsed
#             # print("GOLD PARSED:", gold_parsed)
#         except Exception:
#             gold_parsed = []

#         if len(gold_parsed) == 0:
#             reward = 0.0     

#         try:
#             sol = float(sol)  # to avoid zero division errors
#             gold_parsed = float(gold_parsed)
#         except Exception as e:
#             pass
        

#         tab_struct = completion.split("<table>")[-1].strip().split("</table>")[0].strip()

#         if "```json" in tab_struct:
#                 block = tab_struct.split("```json")[-1].split("```")[0].strip()
#         else:
#             block = tab_struct.replace('\n', '').strip("\n").strip()

#         # print(block)
#         try:
#             jj = json.loads(block, parse_int=str, parse_float=str, parse_constant=str)
#         except:
#             print("Failed to parse JSON")

#         table_json = jj

#         # 2. quick render (bar chart example)
#         fig, ax = plt.subplots(figsize=(2,2))
#         ax.bar(table_json["columns"], [r[1] for r in table_json["rows"]])
#         ax.axis('off')
#         buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)

#         try:
#             pred_type = completion.split("<type>")[-1].strip().split("</type>")[0].strip().lower()
#             if pred_type == c_type.lower():
#                 reward = 1.0
#         except Exception as e:
#             reward = 0.0
        
#         # 3. re-ask
#         checker_out = checker_pipe(image=buf.getvalue(),
#                                    question=question)[0]["answer"]
        
#         # 4. compare
#         try:
#             a, â = float(ans_pred), float(re.findall(r"[-+]?\d*\.?\d+", checker_out)[0])
#             reward = max(0, 1-abs(a-â)/abs(a))
#         except:
#             reward = float(ans_pred.lower() == checker_out.lower())
#         rewards.append(reward)
#     return rewards