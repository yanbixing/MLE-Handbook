#REVISED 

The Mixture of Experts (MoE) concept was invented before the advent of deep neural networks (DNNs). Its application scope is not limited to DNNs or Transformer-based architectures.

## Model Structure

![[moe_illustration.png|400]]  
Fig Ref: [HuggingFace: Mixture of Experts Explained](https://huggingface.co/blog/moe)

The final output of an MoE block is a weighted element-wise sum of outputs from multiple experts, where weights are derived from a gating function, expressed as: $$y = \sum_{i=1}^{n} G(x)_i E_i(x)$$ where:
-  $G(x)_i$: the $i$ -th dimension of the gating (router) function $G$, i.e., the weight assigned to the $i$ -th expert.
- $E_i(x)$: the output of the $i$ -th expert.

In early times, the gating function usually used the standard softmax function: $$G_{\sigma}(x) = \mathrm{Softmax}(x \cdot W_g)$$
However, modern implementations typically use _top-k softmax_: only a small subset (`top-k`) of experts is activated.
- Pro: Far fewer neurons are activated compared to a dense (fully connected) structure, significantly _reducing computation cost_.

An exemplar implementation is Noisy Top-K Gating:
- Noise is introduced to the gate network output as: $$H (x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$
- Only top-k logits are kept, and—similar to a causal mask—inactivated logits are set to −∞-\infty−∞, whose values become 0 after passing through softmax: $$\text{KeepTopK}(v, k)_i = \begin{cases} V_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v, \\ -\infty & \text{otherwise}. \end{cases}$$
- Logits pass through the softmax, transformed into normalized weights for experts: $$G (x) = \text{Softmax}(\text{KeepTopK}(H (x), k))$$
### Typical Configs
- Optimizer: `SGD`
- Number of experts:
    - Small model: `4`–`16`
    - LLM: `32`–`256`
- Gate/routing function: `softmax`
- Experts Activated (`Top-k`): top-2 gating, top-1 gating, or all (standard softmax)
- Depth: 2 linear layers (i.e., 1 hidden layer) 
- Layer Size: `2`–`4` × input dimension
- Input dimension:
    - LLM: `1024`–`12288`
- Capacity Factor (LLM only): number of tokens allowed to process per expert, `1.5`
- Load balance loss weight (LLM only): 0.01 to 0.1, penalty on imbalanced loads.

## Model Comparison

### MoE vs Fully Connected (Dense) Layers:
- **MoE Pros:**
    - **Compute-efficient:**
        - _Lower computation cost_ at _both training and inference time_:
            - Only a small subset of experts is activated each pass, so the activated neurons are far fewer than in the dense layer.
        - _At-parity performance_:
            - Each expert specializes in a smaller scope, so even with smaller size, it can achieve comparable performance with a denser layer.
- **MoE Cons:**
    - **Overall higher memory/storage usage**:
        - Despite each expert's size being small, the number of experts is large, so the cumulative size of all experts exceeds that of a dense layer.            
    - Prone to **overfitting**, especially during **fine-tuning**:
        - _Smaller training data size per expert_: In each pass, only a few experts are "activated," i.e., trained. Equivalently, this means, for each expert, the training data is only a small subset of the overall training data.
        - _Risk of under-trained experts due to imbalanced load_: When imbalanced load happens, less traffic (training samples) is assigned to the less-frequently activated experts, and those experts are under-trained and prone to overfitting.
            - Solution: See the "Frequent Problems" section.
    - (Only for top-k MoE, also the motivation for [[LLM Models (WIP)#DeepSeek MOE|DeepSeek MOE]]): Setting a _`top-k`_ cut-off will result in **losing information** present in dropped experts. I.e., the output may only contain part of the input information, **not comprehensive** enough.
        - **Solution:** [[LLM Models (WIP)#DeepSeek MOE|DeepSeek MOE]], where _one expert is always activated for all tasks_, acting as a **"generalist"** to provide comprehensive/contextual information in all cases.
- **Model Size Comparison:**
	- $\text{Total experts} > \text{Dense layer} > \text{Single expert}$
- Ref: [Reddit: Help Me Understand MOE vs Dense](https://www.reddit.com/r/LocalLLaMA/comments/1l2qv7z/help_me_understand_moe_vs_dense/), [HuggingFace: Mixture of Experts Explained](https://huggingface.co/blog/moe)

## Frequent Problems

### Unbalanced Load

**Problem description:** Some experts may dominate (more frequently activated — easy to underfit), while others are underutilized (less frequently activated — easy to overfit).

_Causes_:
- _Imbalanced domain scope assignment:_
    - During training, symmetry in task allocation can break, causing the decision to rely only on a small subset of experts.
- _Overestimated number of experts_:
    - The number of experts is far larger than required, only a subset of experts is needed.
- _Data imbalance over domain/subtasks:_
    - When the dataset is inherently imbalanced over different domains or subtasks, even if task type assignment is balanced, the data assigned to each expert will be imbalanced.

**Solution:**
- **Auxiliary Loss:** Encourages balanced expert activation by introducing a penalizing term for unbalanced loads in the loss function. Typically implemented as the `aux_loss` hyperparameter.
- **Random Routing:** Randomly selects experts for routing instead of deterministically picking the top-k, usually with probability proportional to the softmax weights.
    - Example: In typical _top-2 routing_ (e.g., Mixtral, DeepSeek), the top-1 expert is chosen deterministically, while the second is randomly sampled based on softmax weight probability.
- **Expert Capacity Capping:** Sets a hard cap on the number of tokens an expert can process. Once exceeded, the expert is "overflowed" and won’t accept new tokens. New tokens are redirected to the remaining available experts.

## FAQ

- What is the typical data type of MoE output in LLMs? Vector or scalar?
    - A vector
- What is the typical gating method of MoE in LLMs?
    - Top-k softmax (e.g., both Mixtral and DeepSeek use top-2 softmax)
- Which LLMs are using MoE (as of 2024)?
    - Using MoE: Mixtral (Dec 2023), DeepSeek (Jan 2024)
    - Not using MoE: LLaMA, Grok, GPT, Gemini, Claude
- What is the typical data type of gate weight in MoEs? Vector or scalar?
    - Scalar (per expert)
- Are there techniques using per-dimension (vector) gating instead of per-expert (scalar) gating?
    - Yes. Gated Linear Units (GLU) generate a vector gate for the input embedding, gating via element-wise multiplication between the gating vector and the input embedding — a form of vector gating. See: [[Llama (WIP)#GELU (Gaussian Error Linear Unit)]]

## Reference

- [HuggingFace: Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [Paper: Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314) — First MoE application in DNNs (2013)