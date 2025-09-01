#REVISED 
## Motivation
[[Chain of Thoughts (CoT) Prompting|CoT]] enables LLMs to perform multi-step reasoning. However, it does not provide access to ==external tools== for obtaining updated or more precise information, which can lead to ==**hallucinations** and **error propagation**==.
 
## Basics
### Definition 
- **ReAct** is a prompting strategy that **enhances [[Chain of Thoughts (CoT) Prompting|CoT]]** reasoning with *action planning*.
    - I.e. **ReAct = CoT + “Action”** (typically involving external tools or APIs).     
- It enables LLMs to leverage external tools during the CoT process, leading to more accurate step-by-step reasoning.
### Illustration
In the original [ReAct paper](https://arxiv.org/pdf/2210.03629?utm_source=chatgpt.com):
- **ReAct =** [[Chain of Thoughts (CoT) Prompting|CoT]] + web search API
	- ![[ReAct_overview.png|300]]
		- Fig Ref: [Coursera: ReAct: Combining reasoning and action](https://www.coursera.org/learn/generative-ai-with-llms/lecture/yCzw0/react-combining-reasoning-and-action#)
- A ReAct example builds on a standard QA with the following components as intermediate steps:
    - **Thought:** A CoT-style reasoning statement that identifies what information is needed.
    - **Action:** An API call to query the needed information inferred in the Thought step.
    - **Observation:** A statement to records the result returned by the Action step.
- Evaluation Benchmarks:
	- HotPotQA: A benchmark for multi-step question answering.
	- FEVER: A benchmark for fact verification using Wikipedia passages.
### Concept Differentiation

#### ReAct vs [[Chain of Thoughts (CoT) Prompting|CoT]]
- Relation: *ReAct* can be viewed as an *enhanced version of CoT*, adding the ability to call external tools during the reasoning process.
- Templating Flexibility:
    - ReAct is like a specialized form of CoT, following a relatively rigid structure with clearly defined stages such as `Thought`, `Action`, and `Observation`.
    - CoT, by contrast, is a broader concept that can be instantiated with more flexible prompting templates.

#### ReAct vs [[Program-aided Language Models (PAL)|PAL]] / [[Retrieval-Augmented Generation (RAG)|RAG]]:
- Orthogonality:
    - ReAct is a framework for enabling CoT to leverage external tools during reasoning.
    - PAL and RAG are standalone techniques with their own workflows and prompting methods, independent of CoT or ReAct.
- Connection:
    - While ReAct itself is not PAL or RAG, it can call external tools with functions such as program execution or information retrieval, thereby achieving functionality similar to PAL or RAG.

## Template Construction

![[ReAct_prompt_structure.png|500]]
(Optional) At the very beginning, usually, an **instruction** is inserted to explicitly ask the LLM to follow the ReAct pattern in the given examples.

**Examples:** The examples provided before the target question typically follow a structured question-and-answer format, with repeated cycles of **thought, action, and observation** as intermediate steps.

- **Question**: Typical example questions as in [[In-context Learning (ICL)|ICL]] technique
- **Repeated cycles** of:
	- **Thought**: The reasoning statement as in typical [[Chain of Thoughts (CoT) Prompting|CoT]] examples.
	- **Action:** A set of predefined operations for LLM to execute, like interact with external data source or output the final answer. E.g. 
		- `Search[entity]`: Retrieve the top X results from a web search API (e.g., some Wikipedia entity or the corresponding page).
		- `Lookup[string]`: Identify and extract relevant information from the retrieved documents (e.g., a sentence containing the target string from a Wikipedia page).
		- `Finish[Answer]`: Conclude the task and return the final answer to the user.
	- **Observation**: The result of the action, such as retrieved info or a calculation result, which is then fed back into the prompt in text format as context for subsequent token generation.
- **Answer**: The final completion, typically output via the `Finish[Answer]` Action.
![[ReAct_vs_CoT_vs_Action-Only.png|700]]
Fig Ref: [Paper: ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

Ref: [Coursera: ReAct: Combining reasoning and action](https://www.coursera.org/learn/generative-ai-with-llms/lecture/yCzw0/react-combining-reasoning-and-action#)

## Orchestration Process

ReAct needs to be realized with an orchestration framework:
- Actions are predefined and mapped to corresponding APIs or tools.
    - For example: `Search`, `Calculator`, `PythonExec`, `SQLQuery`, `GetWeather`.
- _During generation_, the orchestrator continuously monitors the model’s output.
- When a predefined action token is detected, the orchestrator _pauses generation_, _executes the associated API call_, and collects the result.
- The result is then _injected back into the reasoning process_ as an Observation, _appended to the already-generated content_.
    - Just like previously generated tokens, the API result also serves as context for generating subsequent tokens.
- The LLM consumes the API result in a new “pre-fill” stage.
- The LLM then resumes the regular next-token generation process.

Clarifications:
- The API call occurs _during generation_—i.e., it interrupts the output stream, rather than waiting until the full answer (e.g., an `<EOS>` token) is produced.
- This design aligns with the nature of autoregressive generation: since tokens are generated one at a time, not all at once, pausing mid-generation does not break the LLM’s token generation process.
- From the perspective of next-token prediction:
    - API-returned results have no difference from tokens already generated by the model—both function as context for predicting what comes next.
    - For example, in a standard QA task, the question is also not generated by the model.
    - Similarly, API output just needs an additional pre-fill stage, then the LLM can just consume it as other types of context like the question, instructions, provided ICL examples or already-generated tokens.