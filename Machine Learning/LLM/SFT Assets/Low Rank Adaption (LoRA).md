#REVISED 
## Formula

$$W = W^{\text{frozen}} + \alpha A B$$ where:
- $\alpha$ is a hyperparameter that controls the scaling of the tunable component.
- $A$ and $B$ are trainable low-rank matrices.
## Algorithm
![[LoRa_illustration.png|600]]
Fig Ref: [Coursera: GenAI with LLM - PEFT techniques 1: LoRA](https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora#)

**Algorithm Description**:
- Freeze the model's self-attention layers.
    - Other components, like feed-forward layers, are typically frozen as well.
- Add a **trainable matrix** of the same shape to the self-attention weights.
    - Such an operation is equivalent to making the self-attention weights trainable.
- Instead of filling the trainable matrix with independent trainable weights as elements, it is composed as **a product of two low-rank matrices**, each with independent trainable weights as elements.
    - Such a design can significantly _reduce the number of independent trainable weights_. E.g.
	    - ![[LoRA_vs_full_param_fine-tuning_comparision.png|600]]
		    - Fig Ref: [Coursera: GenAI with LLM - PEFT techniques 1: LoRA](https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora#)

**Interpretation**: Such a method allows us to:
- Make ALL weights in the attention layer tunable,
- While significantly reducing the number of trainable parameters, and thus _reducing_ training-time _computational cost_ and _memory usage_ (since each parameter requires additional memory for gradients and optimizer states).

**Note**:
- Inference time: Using LoRA during SFT has no influence on inference-time FLOPs or memory, as LoRA weights are added to model weights.
- Storage:
    - A single LoRA adapter combined with a base model requires more storage than a fully fine-tuned model.
    - However, storing multiple LoRA adapters with one shared base model requires far less storage than storing multiple fully fine-tuned copies.

In sum, **pro of LoRA** vs full fine-tuning (FFT):
- Significantly reduce _computational cost_ and _memory consumption_ at training (SFT) time.
- No impact on computational cost and memory consumption at inference time.
- Significantly *reduce storage space* when when maintaining _multiple fine-tuned copies_.

## Key Hyperparams

- **Rank** `r`: Rank of the low-rank matrices $A$ and $B$, which determines the number of trainable parameters in LoRA.
    - Typical value: `8`
    - Behavior: A **higher rank** does **NOT necessarily improve performance**—performance gradually saturates/plateaus.
        - ![[LoRA_rank_plateau.png|600]]
            - Fig Ref: [Coursera: GenAI with LLM - PEFT techniques 1: LoRA](https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora#)

- **Scaling factor** `alpha`: Scales the output of the LoRA component; internally divided by `r` during execution.
    - Typical value: `2 × r = 16` (resulting in an effective scaling factor of `2`)
    - Behavior: The **higher the `alpha`**, the _greater the influence of LoRA_ on the original weights and the higher the variance error (i.e., **overfitting**).

- **Initialization** `init_lora_weights`: Initialization method for matrices $A$ and $B$.
    - **Typical values**:
        - **B**: Zero initialization (i.e., **all zeros**)
        - **A**: Non-zero initialization (e.g., Gaussian, Kaiming-norm, etc.)

- **`target_modules`**: Specifies which parts of the transformer model LoRA is applied to (e.g., projection matrices of `q`, `v`, `k`, `o`).
    - **Typical value**: `["q_proj", "v_proj"]` — i.e., _apply LoRA only to the **query** and **value** projection matrices_ in the _attention block_.

Reference: [HuggingFace: LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

## Deployment
- You don’t need to store the entire model like in FFT—only the LoRA adapter matrices are sufficient. When switching tasks, simply switch the LoRA matrices.
  ![[LoRA_switch_task.png|600]]
	- Fig Ref: [Coursera: GenAI with LLM - PEFT techniques 1: LoRA](https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora#)

## FAQs

### Application Target
#### 1. Which transformer components is LoRA typically applied to?
- LoRA is most commonly applied to the **query (`q`) and value (`v`) projection matrices** in **attention** layers.  
- Ref:
    - [Paper: LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685): "only the query and value projection matrices being adapted"
    - [GitHub: LoRA: Low-Rank Adaptation of Large Language Models](https://github.com/microsoft/LoRA): LoRA is applied to `q_proj` and `v_proj`
    - [HuggingFace: LoRA](https://huggingface.co/docs/peft/en/package_reference/lora): `target_modules=["q", "v"]`
    - [GitHub: HuggingFace - PEFT - test_target_parameters.py](https://github.com/huggingface/peft/blob/f3b97c370489ff480d00aaebda86bfbbf927a728/tests/test_target_parameters.py#L69): `"target_modules": ["q_proj", "v_proj"]`

#### 2. Why is LoRA typically ONLY applied to attention layers, but not other layers like FFNs?
- Theoretically, LoRA can be applied to other parts. But it is not as efficient as applying it to attention layers:
    - Attention layers have _fewer parameters_ than FFNs but are _equally or more critical_ for performance.
    - Thus, applying LoRA to attention yields a better performance-to-resource tradeoff.
    - Ref:
        - [Paper: A Note on LoRA](https://arxiv.org/html/2404.05086v1): "Nonetheless, considering the additional memory demands of LoRA, attention-based LoRA typically offers greater efficacy within memory constraints."
        - [Paper: Not All Parameters Are Born Equal: Attention Is Mostly What You Need](https://arxiv.org/abs/2010.11859): "The FFN has considerably more parameters than the attention, but they both seem to be equally important for the model’s performance."
        - \[ChatGPT: attention vs FFN layer parameter number in LLaMA 7B and 65B\]
            - LLaMA-7B: 2.15B in attention, 4.29B in FFN
            - LLaMA-65B: 21.4B in attention, 28.9B in FFN

#### 3. Inside the attention layer, why is LoRA typically ONLY applied to Q and V, but not K?
- The attention output is a weighted sum of value vectors. Therefore, fine-tuning attention requires tuning both the attention weights ($Q \cdot K$) and the values ($V$):
    - Applying LoRA to $V$ is necessary.
    - Since _attention weights are computed as $Q \cdot K$_, applying LoRA to **either** $Q$ **or** $K$ is sufficient for modifying the distribution.
- *Why prefer Q over K?*
    Interpretation of $Q$ and $K$:
    - $Q$: Controls how a token attends to others—**determines the attention distribution that the token allocates to other tokens.**
    - $K$: Defines how a token represents itself when being attended to—a more static/inherent identity representation.
- Since our goal is to _adjust how each token allocates its attention_, $Q$ is a more intuitive target for tuning.

### Initialization
#### 1. Why must matrix $B$ be initialized to zero?
- To ensure that _at $t = 0$, the network behaves identically to the original network_: $$W_{t=0} = W^{\text{frozen}}_{t=0} + \alpha BA = W^{\text{frozen}}_{t=0} + 0 = W^{\text{frozen}}_{t=0}$$
- Ref: [[Weight Initialization#LoRA]]

#### 2. Why does $B = 0$ NOT lead to gradient vanishing like regular weight zero-initialization?
- According to [[Backpropagation]]: $$\mathbf{grad}^{{n-k}} = \mathbf{grad}^{{n-k+1}} \left[ \left( X^{(n-k+1)} \right)^{(-1)} W^{(n-k+1)} g'(Z^{(n-k)}) X^{{n-k}}  \right]$$the gradient at the current layer depends on:
    - The gradient from the next layer $\mathbf{grad}^{n-k+1}$
    - The weights of the next layer $W^{(n-k+1)}$
- I.e.  [[Gradient Vanishing (WIP)]] occurs when $W^{(n-k+1)}$ is zero.
    
- In LoRA: $W^{(n-k+1)} = W^{{(n-k+1)}, \text{frozen}} + \alpha BA$, 
	- When $B = 0$,  $BA = 0$
	- but: $W^{(n-k+1)} = W^ {{(n-k+1)}, \text{frozen}} + 0 = W^ {{(n-k+1)}, \text{frozen}} \neq 0$
	- I.e. *the **total weight** of the adapted layer is **non-zero**, it previous layers' gradients won't be 0.*
- Thus, initializing $B = 0$ does NOT cause gradient vanishing.
- Ref [[Weight Initialization#LoRA]]

#### 3. Why must matrix $A$ be initialized to non-zero (i.e., why CANNOT both $A$ and $B$ be zero)?
- The gradients for LoRA matrices $A$ and $B$ are: $$\begin{cases} \begin{aligned}\mathbf{grad_A}^{{n-k}} & :=
- \frac{\partial R}{\partial A^{{n-k}}} = \frac{\partial R}{\partial W^{{n-k}}} \frac{\partial W^{{n-k}}}{\partial A^{{n-k}}} = \frac{\partial R}{\partial W^{{n-k}}} \frac{\partial \left( W^\text{frozen} + B^{{n-k}}A^{{n-k}} \right)}{\partial A^{{n-k}}} \\ & = B^{{n-k}}\mathbf{grad}^{{n-k}} 
- \end{aligned} \\ \\ \begin{aligned}\mathbf{grad_B}^{{n-k}} & = A^{{n-k}}\mathbf{grad}^{{n-k}} \end{aligned} \end{cases}$$
- If _both $A^{{n-k}}_{t=0} = 0$ and $B^{{n-k}}_{t=0} = 0$ at $t = 0$ _:
	- Then $\mathbf{grad_A}^ {{n-k}}_{t=0} = 0$ and $\mathbf{grad_B}^ {{n-k}}_{t=0}=0$ at $t=0$,
	- Then, at $t = 1$, both $A$ and $B$ will still be 0, i.e., **parameter updates will get stuck**.
- However, if only $B^{{n-k}}_{t=0} = 0$ and $A^{{n-k}}_{t=0} \neq 0$ at $t=0$:
	- Then, at $t=0$, $\mathbf{grad}_A^{n-k}$ still $= 0$, but $\mathbf{grad}_B^{n-k} \neq 0$ 
	- Then, at $t=0$, $B$ becomes non-zero too, i.e., from $t = 1$, both $A$ and $B$ can be updated normally.
- Ref:  [[Weight Initialization#LORA]]

## Reference:
- [Paper: LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [Coursera: GenAI with LLM - PEFT techniques 1: LoRA](https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora#)