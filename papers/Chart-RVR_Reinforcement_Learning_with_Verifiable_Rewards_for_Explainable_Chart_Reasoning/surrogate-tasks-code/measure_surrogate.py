import os
import json



def comp_table(gt_table, pred_table):
    try:
        gt = json.loads(gt_table, parse_int=str, parse_float=str, parse_constant=str)
    except:
        return 0
    try:
        pred = json.loads(pred_table.replace("'", '"'), parse_int=str, parse_float=str, parse_constant=str)
    except:
        return 1.0
    

    err = 0
    cnt = 0
    
    try:
        for r1, r2 in zip(gt['rows'], pred['rows']):
            for ent in range(len(r1)): 
                cnt+=1
                if r1[ent].strip().lower() != r2[ent].strip().lower():
                    err+=1
            
            # for c1, c2 in zip(r1, r2):
            #     breakpoint()
            #     cnt+=1
            #     if str(c1.lower()).strip() != str(c2.lower()).strip():
            #         err+=1
        
    except:
        err += 1.0
    
    # cerr = 0
    # cnt = 0
    # try:
    #     for c1, c2 in zip(gt['columns'], pred['columns']):
    #         for ent in range(len(c1)): 
    #             cnt+=1
    #             if c1[ent].strip().lower() != c2[ent].strip().lower():
    #                 cerr+=1
            
    #         # for c1, c2 in zip(r1, r2):
    #         #     breakpoint()
    #         #     cnt+=1
    #         #     if str(c1.lower()).strip() != str(c2.lower()).strip():
    #         #         err+=1
        
    #     cerr+= cerr/(cnt+1e-6) 
    # except:
    #     cerr+= cerr/(cnt+1e-6) 

    return err/(cnt+1e-6) #+ cerr
    


def main():
    dsets = ["chartqa-src", "plotqa", "chartfc", "evochart", "chartqapro", "chartbench"]
    for dset in dsets:
        print("Processing dataset:", dset)
        with open("gt_classify/"+dset+"-2026.json", "r") as f:
            gt_info = json.load(f)

        with open("logs/cot-hard-"+dset+".out", "r") as f:
            cot_info = f.read()
        
        with open("logs/grpo-hard-"+dset+".out", "r") as f:
            grpo_info = f.read()

        with open("logs/sft-hard-"+dset+".out", "r") as f:
            sft_info = f.read()

        cot_info = cot_info.split("#####")
        grpo_info = grpo_info.split("#####")
        sft_info = sft_info.split("#####")

        grpo_type_acc = 0
        sft_type_acc = 0
        cot_type_acc = 0
        grpo_table_err = 0
        sft_table_err = 0
        cot_table_err = 0 


        for entity in range(len(gt_info)):
            # Capture GT
            gt_table = gt_info[entity].split("```json")[-1].split("```")[0].strip()
            gt_type = gt_info[entity].split("### Type:")[-1].strip()
            gt_rsn = gt_info[entity].split("### Reasoning: ")[-1].strip().split("### Type:")[0].strip()

            # print(gt_type, gt_rsn, gt_table)
            
            # Capture COT
            cot_type = cot_info[entity].split("<type>")[-1].split("</type>")[0].strip()
            cot_table = cot_info[entity].split("<table>")[-1].split("</table>")[0].strip()
            cot_rsn = cot_info[entity].split("</table>")[-1].strip()

            # Capture GRPO
            grpo_type = grpo_info[entity].split("<type>")[-1].split("</type>")[0].strip()
            grpo_table = grpo_info[entity].split("<table>")[-1].split("</table>")[0].strip()
            grpo_rsn = grpo_info[entity].split("</table>")[-1].strip()

            # Capture SFT
            sft_type = sft_info[entity].split("<type>")[-1].split("</type>")[0].strip()
            sft_table = sft_info[entity].split("<table>")[-1].split("</table>")[0].strip()
            sft_rsn = sft_info[entity].split("</table>")[-1].strip()

            cot_type_acc += int(gt_type.lower() == cot_type.lower())
            grpo_type_acc += int(gt_type.lower() == grpo_type.lower())
            sft_type_acc += int(gt_type.lower() == sft_type.lower())

            cot_table_err += comp_table(gt_table,cot_table)
            grpo_table_err += comp_table(gt_table,grpo_table)
            sft_table_err += comp_table(gt_table,sft_table)


        print(f"COT Type Acc: {cot_type_acc}/{len(gt_info)} = {cot_type_acc/len(gt_info):.2f}")
        print(f"SFT Type Acc: {sft_type_acc}/{len(gt_info)} = {sft_type_acc/len(gt_info):.2f}")
        print(f"GRPO Type Acc: {grpo_type_acc}/{len(gt_info)} = {grpo_type_acc/len(gt_info):.2f}")

        print(f"COT Table Err: {cot_table_err}/{len(gt_info)} = {cot_table_err/len(gt_info):.2f}")
        print(f"SFT Table Err: {sft_table_err}/{len(gt_info)} = {sft_table_err/len(gt_info):.2f}")
        print(f"GRPO Table Err: {grpo_table_err}/{len(gt_info)} = {grpo_table_err/len(gt_info):.2f}")


    
main()