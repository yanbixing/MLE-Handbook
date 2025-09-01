#REVISED 
## Definition 
- Unlike vanilla [[Chain of Thoughts (CoT) Prompting|CoT]], which requires reasoning examples to be manually crafted, 
- Auto-CoT leverages the [[Zero-Shot CoT]] technique to **automatically generate representative reasoning examples** from a dataset. These representative reasoning examples are then used for CoT prompting at inference time.

## Algorithm :

1. Given a dataset, compute sentence embeddings of all questions (e.g., using Sentence-BERT).    
2. *Cluster the questions into $k$ clusters*.
3. Initialize a placeholder $R_k$ to hold $k$ question-rationale-answer $(q,r,a)$ pairs to be generated in the next step. Each will serve as the representative reasoning example for its cluster.
4. For each cluster $i$:
    - *Sort questions by their distance to the cluster centroid*, in ascending order.
    - From closest-to-centroid to farthest:
        - Generate rationale $r_j$ for the QA pair $(q_j, a_j)$.
        - *If $r_j$ satisfies certain predefined criteria* (e.g., < 60 words, < 5 steps), *select it as the it representative reasoning example for this cluster* and the corresponding $(q_j, r_j, a_j)$ pair will be added to the representative sample dataset.
        - Otherwise, continue to the next question.        
    - This yields a reasoning example dataset:  $R_k = \{(q_1,r_1,a_1) \dots (q_k,r_k,a_k)\}$
5. At inference time, *integrate ALL examples in $R_k$ into the prompt as CoT exemplars.

Illustration:
![[auto_CoT_illustration.png|600]]
Fig Ref: [Paper: Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)

## References
- [Paper: Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)