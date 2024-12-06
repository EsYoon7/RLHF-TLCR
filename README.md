# RLHF-TLCR
[ACL'24 Findings] Official code for "TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback"

This repository provides the official implementation of our ICLR 2024 paper:
> TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback   
> Authors: Eunseop Yoon*, Hee Suk Yoon*, SooHwan Eom*, Gunsoo Han, Daniel Wontae Nam, Daejin Jo, Kyoung-Woon On, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo

The implementation is built upon [ReMax](https://github.com/liziniu/ReMax).


## Installation


The Python environment can be set up using Anaconda with the provided `environment.yml` file.

```
conda env create -f environment.yml
conda activate llm
```


## Running Experiments


#### Step 1 Supervised Finetuning

```
cd step1_supervised_finetuning

# Llama2(7B)
bash training_scripts/llama2/run_llama2_7b.sh
```

#### Step 2 Reward Model Learning

```
cd step2_reward_model_finetuning

# Seqeuntial reward model 
bash training_scripts/llama2/run_llama2_7b.sh

# Token discriminator model  
bash training_scripts/llama2/run_llama2_discriminator.sh
```

#### Step 3 RLHF with token level reward model (TCLR)

```
cd step3_rlhf_finetuning_tlcr

# Llama2 (7B)
bash training_scripts/llama2/run_llama2_tlcr.sh
```


#### Modified Dataset
You can download the modified dataset we used from [Google Drive](https://drive.google.com/drive/folders/1JzzGe9RPg34NZjn_9z6rWCe-__Gqwvub?usp=sharing)
It includes 40% of the HH-RLHF dataset.

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions), Institute of Information communications Technology Planning Evaluation (IITP) grant funded by the Korea government(MSIT) [RS-2021-II212068, Artificial Intelligence Innovation Hub (Seoul National University)] and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)(No. RS-2019-II190079, Artificial Intelligence Graduate School Program, Korea University).


Also, we thank the authors of the [ReMax](https://github.com/liziniu/ReMax) for their open-source contributions and their assistance with the sharing rlhf experience


## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{
Yoon2024tlcr,
title={{TLCR}: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback},
author={Eunseop Yoon, Hee Suk Yoon, SooHwan Eom, Gunsoo Han, Daniel Wontae Nam, Daejin Jo, Kyoung-Woon On, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo },
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics Findings},
year={2024},
}
```

## Contact
If you have any questions, please feel free to email esyoon97@kaist.ac.kr