#!/bin/bash

# run data whitening with 20% compression ratio
# python SVDMoE.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
# my step 1
# python SVDMoE.py --model 'OLMoE' --step 0 --ratio 0.1 --whitening_nsamples 32 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path './outfile/'.
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDMoE.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource


# # evaluate the perplexity of llama_7b_whitening_0.2.pt
python SVDMoE.py --step 4 --model_path outfile/OLMoE_SVD_only_0.9.pt

# # evaluate the efficiency of llama_7b_whitening_0.2.pt
# python SVDMoE.py --step 5 --model_path whitening/jeffwan_llama_7b_hf_whitening_0.8.pt


