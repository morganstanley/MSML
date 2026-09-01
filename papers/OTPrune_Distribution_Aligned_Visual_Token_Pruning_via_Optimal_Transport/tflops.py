def find_flop(seq_len,hidden_state_dim,intermediate_ffn,total_layers):
    n = seq_len
    d = hidden_state_dim
    m = intermediate_ffn
    T = total_layers
    flop = T *( (4 * n * (d**2)) + (2 * (n**2 * d)) + (2 * n * d * m) ) 
    return flop

def find_tflop_ratio(layer, subset_ratio, sequence_len, total_layers, total_visual, hidden_state_dim=4096, intermediate_ffn=11008  ): 
    # layer number 1, ..., T
    n = sequence_len
    d = hidden_state_dim
    m = intermediate_ffn
    T = total_layers
    r = subset_ratio
    total_flop = find_flop(n,d,m,T)
    reduced_visual_tokens = int(round((1-r)*total_visual))
    #print(f"removing {reduced_visual_tokens} tokens")
    updated_seq_len = sequence_len - reduced_visual_tokens
    shorter_flop = find_flop(n,d,m,layer) + find_flop(updated_seq_len,d,m,T - layer)
    flop_ratio = ( (float(shorter_flop)/float(total_flop)))*100.0 
    return shorter_flop/(10**12), flop_ratio,total_flop

def llava_tflops(model_name='llava_1.5', model_size='7b', data_type = 'image'):

    if '7b' in model_size:
        v =  576
        d = 4096 
        m = 11008
        T = 32 # layers 
    elif '13b' in model_size:
        v = 576
        d = 5120
        m = 13824
        T = 40 # layers
    else: 
        raise ValueError("Mode details not found. Add model details.")

    if 'image' in data_type.lower():
        # List of baseline names for V1 experiment 
        # coco2017_cap_val, flickr30k_test, gqa, mmbench_en_dev,mme,mmmu_val,nocaps_val, ok_vqa_val2014,pope,scienceqa_img,seedbench
        text_toknes_list = [15,15,23,83,25,113,15,39,21,91,57]
        if 'llava_1.6' in model_name.lower():
            visual_tokens_list = [2225,1922,2239,1927,2108,1882,2292,2249,2250,1598,2116] 
    
    elif 'video' in data_type.lower():
        # Video Benchmarks [activitynet, Seedbench, VideoChatGPT, NextQA, Egoschema]
        text_toknes_list = [28,63,21,38,196]
        v = 8*144 #8 frames
    else:
        raise ValueError("Data modality not supported .")


    array_flop_ratio = []
    array_pruned_flop = []
    array_full_flop = []
    
    for i, text_tokens in enumerate(text_toknes_list):
        if ('llava_1.6' in model_name.lower()) and ('image' in data_type.lower()):
            v = visual_tokens_list[i]
        
        visual_tokens = v
        n =  text_tokens + visual_tokens  
        if 'image' in data_type.lower():
            parameters = {
                'DivPrune' : {'n': n, 'k':0 , 'r':0.098 },
            }
        else:
            parameters = {
                'DivPrune' : {'n': n, 'k':0 , 'r':0.1 },
            }      
        for baseline in parameters:
            K = parameters[baseline]['k']
            r = parameters[baseline]['r']
            n = parameters[baseline]['n'] 

            pruned_flop, flop_ratio, full_flop = find_tflop_ratio(K, r, n, T, v,d,m)
            array_flop_ratio.append(flop_ratio)
            array_pruned_flop.append(pruned_flop)
            array_full_flop.append(full_flop)
   
    print(f"avg full TFLOPs: {((sum(array_full_flop)/float(len(array_full_flop)))/(1.e12)):.4f}")
    print(f"avg pruned TFLOPs: {(sum(array_pruned_flop)/float(len(array_pruned_flop))):.4f}")
    print(f"avg ratio TFLOPs: {(sum(array_flop_ratio)/float(len(array_flop_ratio))):.4f}")


if __name__ == "__main__":
    print("=== LLaVA 1.5 7b - image understanding ===")
    llava_tflops(model_name='llava_1.5', model_size='7b', data_type = 'image')
    print("=== LLaVA 1.5 13b - image understanding ===")
    llava_tflops(model_name='llava_1.5', model_size='13b', data_type = 'image')
    print("=== LLaVA 1.6 7b - image understanding ===")
    llava_tflops(model_name='llava_1.6', model_size='7b', data_type = 'image')
    print("=== LLaVA 1.6 7b video understanding ===")
    llava_tflops(model_name='llava_1.6', model_size='7b', data_type = 'video')