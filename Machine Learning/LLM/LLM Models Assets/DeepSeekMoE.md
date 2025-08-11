#REVISED 

In DeepSeek-V3, the typical dense Feed-Forward Network (FFN) block is replaced by a Mixture-of-Experts (MoE) block. And to overcome limitations in conventional MoE architecture, DeepSeek introduces an enhanced design, DeepSeekMOE, illustrated below:
![[DeepSeek_MOE_to_DeepSeekMOE.png|600]]
Fig Ref: [Paper: DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

## Recap: Standard MOE

Instead of a single dense FFN, MoE uses multiple parallel FFNs (“experts”). Each expert is typically smaller than the full FFN.

- Router:    
    - Computes a scalar affinity score between the input embedding and each expert (at the token–expert level, not per dimension).
    - As its name suggests, the router acts as an assignment dispatcher: only the top-k experts with the highest scores are activated and process the input embedding.
    - _Non-activated experts are skipped (no computation is executed) to save computational resources_.
- MOE output:
    - Outputs from the activated experts are combined in a _weighted_ **element-wise superposition** manner, where weights are derived from the affinity scores, typically via a softmax operation.

Workflow:
1. The MoE block receives embedding from the attention block.
2. The router computes affinity scores between the embedding and experts.
3. The input embedding is fed to the top-k experts with the highest affinity scores.
4. Outputs from these experts are weighted and element-wise superposed together as the final MoE block output.

Problem:
- Since each expert captures only a subset of aspects of the input embedding's information, theoretically, full coverage of the input information requires all experts to be active.
- In other words, the top-k expert selection approach only captures the most important aspects; _other aspects of information from non-activated experts are lost_, resulting in a _biased and incomplete representation_ of the input embedding, degrading the model’s performance.

Ref: [[Mixture of Experts (MoE)]]

## DeepSeekMOE

DeepSeek’s design improves both model structure and engineering efficiency.
### Model Structure

#### Shared Expert - A Generalist

To address the issue of lacking comprehensive information in traditional MoE, DeepSeekMOE introduces a **shared expert**. In contrast to the _optionally activated_ _specialized_ experts in traditional MoE, the shared expert:
- Is _always activated_.
- Acts as a **"generalist"**, providing comprehensive **contextual and commonsense knowledge**.

Personal interpretation:
- Conceptually, the shared expert functions somewhat similarly to a skip connection — it allows information in the embedding to pass forward with less distortion, facilitating more efficient gradient flow.
- While it doesn’t fully bypass the transformation layer like a skip connection, the generalist's distortion could be milder than that of specialists (as it is required to preserve more comprehensive information), which leads to a less distorted (attenuated) gradient backpropagation, mitigating the [[Gradient Vanishing (WIP)]] problem.

#### Fine-Grained MoE - More Specialized Experts

With the shared expert providing comprehensive context information, the specialist experts are allowed to be more narrowly focused, i.e., more specialized in fine-grained domains, without worrying about losing comprehensive information.

Thus, the fine-grained MoE is designed as:
- *Higher total expert number*:
    - Each expert covers a smaller, more specific scope, enhancing the specialization of experts in their finer domain.
    - Finer domain segmentation also means finer domains are decoupled; the selection of the most relevant domain (expert) is more flexible and accurate. See the demonstration below.
- *Higher `top-k` value*:
    - More experts are activated, ensuring the scope coverage of the output embedding is not reduced.
- *Smaller individual expert size*:
    - Offsets the potential computation burden caused by an increased number of activated experts, ensuring training and inference costs don’t increase much.
    - Offsets the issue of smaller training sample sizes per expert caused by an increased total number of experts, ensuring the experts are not overfitting or undertrained.
        

Demonstration of decoupled finer domain:
- Assuming there are 6 domains `D1` … `D6`. For a given sample, `D1` and `D3` are the most important.
- Standard MoE:
    - `expert_number`: `3`
        - Expert–domain mapping: `E1:[D1, D2]`, `E2:[D3, D4]`, `E3:[D5, D6]`
    - `top-k`: `1`
        - Domain coverage of output embedding: either `E1:[D1, D2]` or `E2:[D3, D4]`; i.e., one important domain is missed.
- Fine-grained MoE:
    - `expert_number`: `6`
        - Expert–domain mapping: `E1:D1, E2:D2, …, E6:D6`, domains are more decoupled.
    - `top-k`: `2`
        - Domain coverage of output embedding: `E1:D1` and `E3:D3`, i.e., covering both important domains.

**Summary**: The DeepSeek MoE design allows _finer-grained specialized experts_ _without losing context or generalization ability_, enabling the model to _capture more complex patterns_ and _achieve higher prediction accuracy_ _without increasing computational cost_.

### Engineering Efficiency

#### Device Imbalance

In addition to addressing expert imbalance as in standard MoE designs, DeepSeekMOE also tackles **device imbalance** to improve hardware utilization.
- Recap - Expert Imbalance:
    - Definition: Some experts are activated and trained far more frequently than others.
    - Impact:
        - Degraded expert performance:
            - _Undertrained experts_: infrequently trained experts cannot learn well.
            - Over-scoped experts: overly frequently trained experts cover extra scope, reducing their specialization and resulting in degraded performance.
        - Leads to _device imbalance_ and lower computation efficiency.

- Device Imbalance:
    - Definition: Experts are distributed across multiple GPUs. The load of different GPUs can be uneven, i.e., some GPUs are overloaded while others remain idle.
    - Causes:
        - Expert imbalance: uneven load distribution over different experts can cause uneven load distribution over corresponding GPUs.            
        - Correlated expert assignment: even when load is balanced at the expert level, if experts assigned to the same GPU are correlated — often activated or skipped together — the load distribution over GPUs becomes imbalanced.
    - Impact: lower hardware utilization efficiency, wasted computational resources, longer training time, and higher cost.

**Solution**: **Add penalty terms** for both expert load and device load **in the loss function**.
- Interpretation: The more an expert or device is used, the higher the penalty for activating it — encouraging the model to diversify selection toward less-used experts and devices.

## References:
- [Paper: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)