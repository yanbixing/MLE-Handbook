#REVISED 
## Basics

**Definition**: Fine-tune a pre-trained model by providing examples that demonstrate how the model should respond to a specific instruction.
- Most LLM "fine-tuning" refers to instruction fine-tuning by default.

**Key Characteristics**:
- Pro: Usually, fine-tuning an LLM for a single task only **requires** very few examples, 300–500 (significantly fewer than typical DNN fine-tuning).
- Con: Prone to [[Catastrophic Forgetting]], i.e., performance on other tasks significantly drops, losing generalizability.

## Context

- Motivation (Cons of [[In-context Learning (ICL)]]):    
    - LLMs have limited context windows, restricting the number of examples that can be used at inference time.
    - [[Prompt Engineering]] techniques like ICL often underperform on smaller models due to their limited generalization capabilities.
        
- **Concept Differentiate**: Instruction Fine-tuning vs. [[In-context Learning (ICL)]]
    - [[In-context Learning (ICL)|In-Context Learning]]:
        - A [[Prompt Engineering]] technique, i.e., revising the input prompt during **inference time**.
        - Usually **needs** to provide additional prompt-completion **examples** in the input prompt (except zero-shot, which also provides instruction but no example).
    - Instruction Fine-tuning:
        - An [[SFT]] technique, specifically, revising the training data during **training (fine-tuning) time**.
        - Only **includes** explicit **instructions** in the **template**. Usually does not include additional prompt-completion examples in the template.

## Method

### Overview

- High-level Description: With a hand-crafted template, inject a specific "instruction" slice into the prompt-completion pair data used for model fine-tuning. E.g.,
    - Goal: Improve the model's performance on a text summarization task.
    - Approach: Explicitly include the instruction "summarize the text" in the templates.
- Interpretation: During training (fine-tuning), the model learns to attend to the instruction and align its output with the expected format, thus **improving** its ability to follow similar instructions at inference time.
    
    - ![[instruction_fine_tuning_illustration.png|600]]
        
        - Fig Ref: [Coursera: GenAI with LLM - Instruction fine-tuning](https://www.coursera.org/learn/generative-ai-with-llms/lecture/exyNC/instruction-fine-tuning#)
            
- Computing Resources:
    - Instruction fine-tuning is primarily a _data engineering_ technique, not affecting the training process. Therefore:
        - Itself has minimal impact on computing resources—aside from slight increases in token count.
        - It is compatible with other efficient computing optimization techniques like quantization and PEFT.

### Process
#### Instruction Dataset Preparation

- Find a dataset _related to your task_.
- _Compose a template_ based on your task and the dataset, _explicitly including relevant instruction into the template._
- Transform the data with the template.

**Note:** Usually, the same dataset can be customized for different kinds of tasks with different templates/instructions.
- E.g., the Amazon review dataset can be customized for:
    - Review rating prediction (Regression task, 1st template in figure below)
    - Review generation based on rating (LM task, 2nd template in figure below)
    - Review summarization (LM task, 3rd template in figure below)
    - ![[a_same_dataset_with_different_instructions.png|600]]
    
	    - Fig Ref: [Coursera: GenAI with LLM - Instruction fine-tuning](https://www.coursera.org/learn/generative-ai-with-llms/lecture/exyNC/instruction-fine-tuning#)

#### SFT with Instruction Dataset

![[LLM_fine-tuning_loss.png|600]]  
Fig Ref: [Coursera: GenAI with LLM - Instruction fine-tuning](https://www.coursera.org/learn/generative-ai-with-llms/lecture/exyNC/instruction-fine-tuning#)

- Instruction fine-tuning is primarily a _data engineering_ technique, not affecting the training process. I.e.,
    - Instruction fine-tuning's training process is still a normal SFT LM task, i.e., predicting the next token with the prompt (Q) and previously generated completion (A) slice as context (Q part is not a prediction target and thus not considered in the loss term).

## Problem: Catastrophic Forgetting
![[Catastrophic Forgetting]]

### How about Multi-task Instruction Fine-tuning
- **Cons** of Multi-task instruction fine-tuning vs. single-task instruction fine-tuning: Multi-task usually needs a lot of data—typically 50,000 to 100,000. (Single-task needs just 300–500.)
## Appendix
### Relevant Tool - FLAN
![[FLAN]]