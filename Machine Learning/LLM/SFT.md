#REVISED 
# Basics:

## Definition

Full Name: Supervised Fine-tuning

Motivation: Similar to traditional fine-tuning, the goal of SFT for large language models (LLMs) is to "adapt" (improve its performance) the model to a specific domain or task.

**Key Characteristics of LLM's SFT:**
- LLMs can often be fine-tuned with relatively **few samples (hundreds to a few thousand)**.
- This substantially reduces the amount of manual labeling required for LLM fine-tuning.
## Concept Differentiation: 
### Fine-tuning vs SFT vs Instruction Fine Tuning

- Fine-tuning (FT)
	- The broadest concept among the three, generally referring to the process of adapting a pre-trained model to a specific domain or task by retraining its parameters.
	- Originates from transfer learning and is not exclusive to LLMs.
	- Vs SFT: FT can theoretically be supervised or unsupervised, though supervised approaches are more common in practice.

- SFT (Supervised Fine-Tuning): 
	- "Supervised" can refer either to 
		- the classical supervised paradigm where input features and output labels are explicitly distinct, 
		- or self-supervised paradigms such as masked language modeling (MLM) or next-token prediction (LM), where the same data (e.g., tokens) can serve as both input and output labels, originating from the same source.
	- Like FT, term "SFT" predates LLMs and is not limited to them.

- Instruction Fine Tuning (IFT):
	- A data engineering approach that builds on SFT.
	- Vs SFT: Instead of using raw domain-specific documents directly, the data is transformed using templates. These templates **explicitly describe the task in natural language instructions.**
	- Vs SFT: The method became popular with the emergence of LLMs, as it *relies on LLM's* strong generalization capabilities — more specifically, the *ability* to *understand and follow instructions.*
	- Connection with other methods:
		- Vs  [[In-context Learning (ICL)]]: Instruction Fine-Tuning is inspired by [In-context Learning (ICL)]. Instead of solely employing LLM's ability to understand and follow instructions as in ICL, *Instruction Fine-Tuning* optimizes both the prompt template and model parameters, i.e. like:
			- *[[Prompt Engineering]] ([[In-context Learning (ICL)]]) + SFT*
		- Vs [[Soft Prompts (WIP)]]: *Prompt tuning* is inspired by Instruction Fine-Tuning. Instead of manually crafting prompts with human language as in ICL and IFT, soft prompt tuning learns prompts in a model training paradigm, i.e. like 
			- *[[Prompt Engineering]] ([[In-context Learning (ICL)]]) via SFT*
Ref:
- [GeeksForGeeks: Difference between Fine-Tuning, Supervised fine-tuning (SFT) and Instruction Fine-Tuning](https://www.geeksforgeeks.org/artificial-intelligence/difference-between-fine-tuning-supervised-fine-tuning-sft-and-instruction-fine-tuning/)
- [DataScientest: Instruction Tuning: What is fine-tuning?](https://datascientest.com/en/instruction-tuning-what-is-fine-tuning)
- [StackOverflow: Difference between Instruction Tuning vs Non Instruction Tuning Large Language Models](https://stackoverflow.com/questions/76451205/difference-between-instruction-tuning-vs-non-instruction-tuning-large-language-m)"many datasets/benchmarks on the [Hugging Face Hub](https://huggingface.co/datasets)....most of them don't contain any instructions."
- [Reddit: Difference between fine-tuning instruct/non-instruct LLMs](https://www.reddit.com/r/LocalLLaMA/comments/1cfxspr/difference_between_finetuning_instructnoninstruct/) 
- [Reddit: What is the difference between pre-training, fine-tuning, and instruct-tuning exactly?](https://www.reddit.com/r/learnmachinelearning/comments/19f04y3/what_is_the_difference_between_pretraining/)
- [Reddit: Have people stopped saying "fine tuning" in place of "supervised fine tuning?" Or is there some other fine tuning paradigm method out there.](https://www.reddit.com/r/MachineLearning/comments/1ewezs4/d_have_people_stopped_saying_fine_tuning_in_place/)

### Pre-training vs SFT vs RLHF in LLM
- The typical LLM training process follows: **Pre-training → SFT → RLHF**
- **Pre-Training**
	- Training data description: **"unlabelled" text data**
	- Data requirement:  **large volume** of text data.
	- Data structure: "Unlabeled" means the documents are **not required to have a particular structure**, because:
		- All tokens can be used as prediction label with **CLM** or **MLM** method.
		- I.e. The **loss** function, expectationally, considers **all tokens** in the sequence.
	- Task type: CLM or MLM (fundamentally multi-class classification).
	- Loss: Cross-entropy.

- **"Supervised" Fine-Tuning**
	- Training data description: **"labelled" text data**
	- Data requirement: Datasets must contain **high-quality labels** aligned with the target application domain or task. Examples:
	    - Email [[System Design - Text Completion (WIP)|Text Completion]] requires high-quality email text datasets.
	    - [[System Design - ChatBot (WIP)|Chatbot]] tasks require structured Q&A pairs.
	    - [[System Design - Machine Translation (WIP)|Machine Translation]] tasks require multilingual versions of the same document (e.g., `doc_langA` ↔ `doc_langB`).
	- Data structure: Usually (e.g., [[System Design - ChatBot (WIP)|Chatbot]] , [[System Design - Machine Translation (WIP)|Machine Translation]] ), the text data can be explicitly divided into two parts: **context/features** and **labels**. Using the [[System Design - ChatBot (WIP)|Chatbot]] QA example:
		- Only the **label (A)** tokens are used as prediction targets and contribute to the loss.
		- The **feature (Q)** tokens are used purely as context and do **not** contribute to the loss.
	    - Vs pre-training: the task is still **CLM**, but only part of the tokens are included in the loss calculation.
	- Task type: Besides CLM, SFT can also be performed with regression or classification data, though the **LM head** often needs to be adapted accordingly.
	- Loss: Cross-entropy for CLM tasks (e.g. [[System Design - Text Completion (WIP)|Text Completion]]), or other loss functions depending on the specific task.

**RLHF (Alignment Stage)**
- Training data description: Instead of the text data itself, the alignment stage uses external evaluation on the text data.
- Data structure: Typically (e.g., [[Proximal Policy Optimization (PPO) (WIP)|PPO]], [[Direct Policy Optimization (DPO) (WIP)|DPO]]), with text **pairs** as input and **binary pairwise comparison labels** as output. Vs Pre-training and most SFT:
    - These labels serve as the supervision signal for a **reward model**—explicitly trained in PPO or implicitly derived in DPO—rather than directly supervising the LLM.
    - The reward signal is provided at the **document level**, not at the individual token level. I.e., backpropagation is performed per document rather than per token. (Implementation-wise, the reward is only assigned to the last token, as the final token's "joint probability" represents the entire sentence’s probability. See [[Proximal Policy Optimization (PPO) (WIP)#3. FAQs|PPO FAQ]].)
        - **Note:** For [[Proximal Policy Optimization (PPO) (WIP)|PPO]] and [[Direct Policy Optimization (DPO) (WIP)|DPO]], token-level reward signals are an upgraded variant, not the vanilla version, while token-level supervision is unavoidable in pre-training and SFT.
- Task type: RL or classification-like RL task.
- Objective: RL objectives, e.g.:
    - [[Proximal Policy Optimization (PPO) (WIP)|PPO]]: PPO objective (a sum of policy objectives based on rewards/advantages, value functions, and entropy regularization).
    - [[Direct Policy Optimization (DPO) (WIP)|DPO]]: DPO objective (a [[Negative Log-Likelihood (NLL)]] /CE loss on the difference of completion probabilities).

Trivia (冷知识): After SFT (Supervised Fine-Tuning) or PEFT (usually with QA-style datasets), the resulting model is referred to as **"chat" or "instruct" model**—the origin of "Chat" in ChatGPT.
- Chat logs are also considered QA data: the prompt serves as the **Q**, and the response serves as the **A**.

---
# Methods

## Data Engineering

### Baseline: Domain-Specific Data

Simply use raw, high-quality domain-specific text for SFT—the most straightforward and naive approach. We won’t go into detail here.

**Con:** This method does not fully leverage the LLM’s generalization capabilities.
- Solution: [[Instruction Fine-tuning]]

### Instruction Fine-tuning
![[Instruction Fine-tuning]]

## Model Training

### Baseline: Full Fine-Tuning (FFT)

Fine-tune the entire model, which is the most straightforward form of SFT and very similar to pre-training in terms of parameter optimization. We won’t go into detail here.
- **Note:** Selective fine-tuning (i.e., fine-tuning specific _existing_ parts of the model) is theoretically a kind of PEFT. However, in practice, it is closer to full fine-tuning because the parameter optimization process is the same as regular model (pre-)training.
    
**Problems:**
- Computation cost: LLMs are extremely large, making full fine-tuning very expensive.
- [[Catastrophic Forgetting]]: Because LLMs are much more complex than traditional models, fully fine-tuning them on small datasets can lead to more severe catastrophic forgetting.

Solution: [[Parameter Efficient Fine-tuning (PEFT)]]

### Parameter efficient fine-tuning (PEFT)
![[Parameter Efficient Fine-tuning (PEFT)]]

## FAQ

- How do LLMs handle tasks beyond text-based QA, such as numerical computation? Are they still trained using CLM with cross-entropy loss?
	- **Yes.** Surprisingly, even for tasks involving numerical computation, most LLM pre-training and supervised fine-tuning (SFT) phases still rely on language modeling with a cross-entropy loss.
	- To address the limitations of this approach, many techniques have been proposed to improve LLMs’ numeracy (numerical reasoning ability), without any change on the training setup (loss function, model architecture, etc.). For example:
	    - Data engineering:
		    - _Left-padding numbers_ so all numbers have a fixed length. Ref: [Paper: Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://arxiv.org/abs/1909.07940)
		    - _Scientific notation transformation_ (e.g., converting `314.1` → `3141[EXP]2`). Ref: [Paper: Do Language Embeddings Capture Scales?](https://aclanthology.org/2020.findings-emnlp.439.pdf)
		    - These techniques generally work because fixed, structured patterns are empirically easier for language models to learn and predict. 
		    - Ref: [Paper: Representing Numbers in NLP: a Survey and a Vision](https://arxiv.org/abs/2103.13136)
	    - Embedding:
	        - Using deterministic, handcrafted embeddings for numerical values. Ref: [Paper: Methods for Numeracy-Preserving Word Embeddings](https://aclanthology.org/2020.emnlp-main.384.pdf)    
	- Additionally, of course there are methods that improve numeracy by modifying the training objective, for example:
		- Using regression-like, value-based loss functions for number tokens instead of the traditional cross-entropy loss. Ref: [Paper: Methods for Numeracy-Preserving Word Embeddings](https://aclanthology.org/2020.emnlp-main.384.pdf)