#REVISED 
## Motivation

**Problem statement**: (Without explicit guidance, i.e. "zero-shot" mode), LLMs often **fail to produce** outputs in the **desired format**. E.g. 
- A sentiment analysis style prompt expecting a binary output, 
	- ![[zero_shot_expected_output.png|400]]
		- Fig Ref: [Coursera: GenAI with LLM - Instruction fine-tuning](https://www.coursera.org/learn/generative-ai-with-llms/lecture/exyNC/instruction-fine-tuning#)
- But LLM may return a full sentence.
	- ![[zero_shot_actual_output.png|400]]
		- Fig Ref: [Coursera: GenAI with LLM - Instruction fine-tuning](https://www.coursera.org/learn/generative-ai-with-llms/lecture/exyNC/instruction-fine-tuning#)

**Solution**: **[[In-context Learning (ICL)]]** can guid LLM to generate outputs in the *desired format*.
## Basics

### Definition: 
**Definition**: In-Context Learning (ICL) is a [[Prompt Engineering]] technique that improves response quality by *including in the prompt, with example(s)* that illustrate the desired input–output pattern.
- Edge case: "Zero-shot" ICL means no explicit examples, only instructions, are injected to the prompt.
- Alias: ICL is also often referred to as "zero-shot", "one-shot", or "few-shot" "prompting", depending on the number of examples given in the prompt.
    - Ref: [Book: Generative AI System Design Interview - Page 213](https://www.amazon.com/Generative-AI-System-Design-Interview/dp/1736049143)

**Illustration**: By including examples of sentiment analysis outputs that show binary results in the prompt, the LLM can be guided to produce a single binary token as output rather than generating a full sentence.
-  ![[ICL_few_shot_example.png|400]]
	- Fig Ref: [Coursera: GenAI with LLM - Prompting and prompt engineering](https://www.coursera.org/learn/generative-ai-with-llms/lecture/ZVUcF/prompting-and-prompt-engineering)

**Interpretation**: This method *leverages the LLM’s ability to **follow instructions***. By providing explicit input–output examples, the model is guided to generate responses consistent with the demonstrated pattern.
### Concept differentiation
ICL vs [[Instruction Fine-tuning]] (IFT)
- ICL: Example/instruction injection happens _only at inference time_. Model parameters remain unchanged. I.e. A [[Prompt Engineering]] method.
- IFT: Instruction-injection/templating happens during *training (SFT)*. Model weights will be changed.

## Problems

- Hyperparameter tuning limitation: *Number of Examples*
	- Description:
		- The context window is finite, we cannot add unlimited examples.
		- Adding many examples increases computation load during the prefill stage. 
		- In practice, computation time/latency also increases with token count, because:
			- Though the token processing $O(N)$ and attention calculation $O(N^2)$ can be theoretically perfect parallelized,
			- Real hardware's bandwidth/thread is limited and can be saturated.
	- Therefore, the **common tuning practice** is:
		- Add up to *5–6 examples*.
		- *If performance does not improve, fine-tuning the model* is generally recommended.

- ICL **only** guides on **format**
	- Description:
	    - ICL _only regulates the format_ of the output.
	    - The generated answers are still based solely on statistical patterns in the input prompt.
	    - Results can be _inaccurate_ for tasks requiring _complex reasoning, factual precision, or exact computation._*
	- Possible Solutions:
		- [[Chain of Thought (CoT) Prompting|CoT]] – enable LLM to conduct step-by-step reasoning for complex problems.
	    - [[Retrieval-Augmented Generation (RAG)|RAG]] – enable LLM to access external factual knowledge.
	    - [[Program-aided Language Models (PAL)|PAL]] – enable LLM to execute code for accurate computation.