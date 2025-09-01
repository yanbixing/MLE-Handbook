#REVISED 
## Motivation

### Problem Statement
- LLMs generate answers based on token-level statistical patterns, which limits their ability to perform genuine, implicit reasoning like humans.
- As a result, for tasks that are less intuitive and require multi-step reasoning—such as mathematical problems—LLMs often CANNOT carry out the necessary reasoning steps autonomously, leading to incorrect answers.
### Limitation in ICL

In basic [[In-context Learning (ICL)]], _only_ the question and the _final answer_ are provided. While this helps the LLM recognize the expected answer format, it does *NOT demonstrate **how** the answer is derived.*

As a result, the model generates outputs solely from the given context, without an y reasoning process. This often leads to answers that are format-correct but factually wrong.

For example, in the math problem below, the LLM may produce a random number to match the style of the example, but the value is generated from statistical patterns in the question solely, without following the statistical patterns of proper mathematical deduction.

- Prompt
```
(Instruction/Example:)

Q: I have 4 tennis balls, and bought 2 cans of tennis balls, 3 balls per can. How many tennis balls do I have? 
A: 10

(Target question:)

Now consider: The cafeteria has 20 tomatoes at first, used 6 for lunch, and bought 4 more. How many tomatoes does the cafeteria have?
```
- Answer:
```
27
```

### Limitation of Prompt Chaining

**Definition:** Around 2021, researchers observed that LLMs often struggle with complex tasks requiring multiple intermediate steps. One method that can effectively address this challenge is _prompt chaining_, which involves:
- Decomposing a complex task into a sequence of subtasks.
- Ensuring each subtask's prompt explicitly or implicitly depends on the outputs of the previous steps.

**Limitation:**
- Prompt chaining is highly effective for large-scale, multi-step, well-defined tasks—such as iterative planning or drafting research reports—where a single workflow design can be reused repeatedly.
- For relatively simple and ad hoc questions, like the mathematical problem above, prompt chaining can be unnecessarily cumbersome and inefficient as it involve additional human design/analysis efforts.

## Basics

The solution to the above problems is CoT.
### Definition 
- **Chain-of-Thought (CoT)** is a [[Prompt Engineering]] technique designed to improve an LLM’s performance on complex tasks requiring logic, calculation, or decision-making, by *encouraging the model to* mimic human reasoning process and ==**explicitly**== *output its thought process* (i.e. *intermediate steps*).
- Typically, this is *achieved by including examples with ==**explicit step-by-step reasoning process**== within the input prompt*.

### Illustration
By explicitly showing how to map a daily-life question into a mathematical formula step by step, the LLM is guided to follow the same process, thereby improving its accuracy on math problems.

![[CoT_prompting_example_1.png|600]]
![[CoT_prompting_example_2.png|600]]

Fig Ref: [Coursera: Helping LLMs reason and plan with chain-of-thought](https://www.coursera.org/learn/generative-ai-with-llms/lecture/2bQKl/helping-llms-reason-and-plan-with-chain-of-thought#)
### Concept Differentiation:

#### CoT vs ICL:
- **Vanilla ICL:** Provides only the question and the final answer, without showing any explicit step-by-step reasoning.
- **CoT:** Can be seen as an advanced form of ICL. In addition to the question and final answer, the reasoning process that leads to the answer is also included, allowing the model to learn how the solution is derived.
#### CoT vs. Prompt Chaining

- **CoT:** Enhances LLM output using a **single**, *detailed prompt* that *guides the model* to explicitly explain its reasoning process step by step.
	- I.e. CoT extends is an advanced form [[In-context Learning (ICL)]], guiding LLM to conduct the reasoning steps.
    - I.e. The *reasoning process is carried out by the **LLM***.

- **Prompt Chaining:** Improves LLM performance by decomposing a complex task into a series of (i.e., ***multiple**) prompts (rather than one)*, where each step depends on the output of the previous one. The task decomposition is often _conducted by humans_.
	- The technique is typically regarded as a multi-prompt technique, used in LLM-powered application design, rather than a single-prompt engineering method like CoT.
	- For simple questions, applying prompt chaining means _the reasoning process is carried out by the **human**, not the LLM._

Ref: 
- [TechTarget: What is prompt chaining? Definition and benefits](https://www.techtarget.com/searchenterpriseai/definition/prompt-chaining)
- [IBM: prompt chaining](https://www.ibm.com/think/topics/prompt-chaining)

## Problems

- **High manual effort:** Crafting reasoning examples requires significant human labor.
	- Solution:  [[Zero-Shot CoT]] and [[Auto-CoT]] reduce or automate this effort.
- Lack of factual info: Even with CoT, the reasoning process is still driven by token-level statistical inference. This makes it difficult for LLMs to reliably handle tasks that require _factual accuracy_, _precise calculations_, or _structured execution_.
	- Solution: [[ReAct]], a template-based method that extends CoT with action planning. It allows LLMs to leverage external tools with functionalities similar to [[Retrieval-Augmented Generation (RAG)|RAG]] and [[Program-aided Language Models (PAL)|PAL]] during the CoT process.

Ref: [Coursera: Helping LLMs reason and plan with chain-of-thought](https://www.coursera.org/learn/generative-ai-with-llms/lecture/2bQKl/helping-llms-reason-and-plan-with-chain-of-thought#)

## Further Reading
#### Zero-shot CoT
![[Zero-Shot CoT]]
#### Auto-CoT

![[Auto-CoT]]