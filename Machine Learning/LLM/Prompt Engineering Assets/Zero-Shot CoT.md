#REVISED 

**Definition**: An edge case of the CoT technique:
- **Without** providing any explicit question-rationale-answer step-by-step reasoning examples,
- Simply by adding the instruction _“let’s think step by step”_ to the prompt,
- The LLM can be encouraged to generate intermediate reasoning steps before producing the final answer.

**Illustration**
![[zero_shot_CoT.webp|600]]
Fig ref: [PromptingGuide: Chain-of-Thought Prompting](https://www.promptingguide.ai/techniques/cot)

Ref: 
-  [PromptingGuide: Chain-of-Thought Prompting](https://www.promptingguide.ai/techniques/cot)
-  [Paper: Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)