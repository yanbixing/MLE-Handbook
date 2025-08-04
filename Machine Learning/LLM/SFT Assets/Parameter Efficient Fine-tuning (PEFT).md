#REVISED 
## Basics

### Motivation:

Full fine-tuning (FFT) of Large Language Models (LLMs) presents significant challenges due to:
- **Large Model Size:** LLMs are too large, making any full model-level updating resource-intensive.
- **Extra Memory Consumption in Optimization:** Classical fine-tuning requires much more memory than the model size itself.
	- The memory used by components other than weights is typically **12 to 20 times** larger than the weights. E.g.:
		- ![[LLM_training_ram_consumption.png|600]]
			- Fig Ref: [Coursera: GenAI with LLM - Computational Challenges of Training LLMs](https://www.coursera.org/learn/generative-ai-with-llms/lecture/gZArr/computational-challenges-of-training-llms#)

Solution: PEFT
### High-level Description

Instead of updating the entire model, PEFT updates only a small subset of parameters using different strategies:
1. **Selective Fine-Tuning:** Freeze most layers and fine-tune a small subset of existing model parameters (e.g., specific layers or components).
2. **Reparameterization:** Reparameterize model weights using low-rank representations. E.g.:
    - [[Low Rank Adaption (LoRA)]]
3. **Additive Fine-Tuning:** Keep the original model frozen and **add small, trainable components or layers.** E.g.:
    - **Adapters:** Insert new trainable layers (typically inside the encoder/decoder, after attention or feed-forward layers).
    - **Prompt Tuning:** Apply fine-tuning at the prompt level without modifying the model's core architecture, like adding trainable parameters to the prompt embeddings (e.g., [[Soft Prompts (WIP)]]) or adjusting the embedding weights.

**PEFT's Parameter Magnitude:** Typically, only **15–20%** of the original LLM parameters are fine-tuned.

### Pros and Cons
- **Pro**:
    - **Reduce Computing Resources:** Requires fewer computing FLOPs and less memory during training.
        - E.g., PEFT methods are usually trainable on a single GPU.
    - **Reduce Storage Space:** Instead of updating the original model, most PEFT techniques train additional new components—only these smaller new components need to be stored, not a full model copy.
    - **Mitigate [[Catastrophic Forgetting]]:** Since PEFT modifies only a small portion of the model and the change amplitude is small, it is less prone to [[Catastrophic Forgetting]].
    - **Compatibility:** PEFTs are usually method-level approaches and can often be combined with other engineering-level optimization techniques. E.g.:
        - QLoRA = [[Quantization (WIP)]] + [[Low Rank Adaptation (LoRA)]]
- **Con**:
    - **Model Performance:** Full fine-tuning (FFT) usually yields better performance than PEFT.

## Methods 

### LORA
![[Low Rank Adaption (LoRA)]]

### Soft Prompts
![[Soft Prompts (WIP)]]