# References

## Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

![pdf](papers/wei2023chainofthought/2201.11903.pdf)

 * The paper proposes a method called "chain-of-thought" prompting to improve the complex reasoning abilities of large language models.
* The method involves providing a few chain of thought demonstrations as exemplars in prompting.
* Experiments are conducted on three large language models, and the results show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks.
* The empirical gains can be significant, with a PaLM 540B model prompted with just eight chain-of-thought exemplars achieving state-of-the-art accuracy on the GSM8K benchmark of math word problems.
* The paper provides an example of how chain-of-thought prompting works, with the model first generating intermediate reasoning steps before arriving at the final answer.
* The authors suggest that the reasoning abilities emerge naturally in sufficiently large language models, and that the chain-of-thought method is a simple and effective way to elicit this reasoning.
* The paper also includes a discussion of the limitations of the method and potential areas for future research.
* The findings of the paper have implications for the use of large language models in a variety of applications, including education, customer service, and decision-making.
* The paper was presented at the 36th Conference on Neural Information Processing Systems (NeurIPS 2022).

## Yi: Open Foundation Models by 01.AI

![pdf](papers/ai2024yi/2403.04652.pdf)

 * The Yi model family is a series of language and multimodal models with strong multi-dimensional capabilities.
* The models are based on 6B and 34B pretrained language models, and are extended to chat models, 200K long context models, depth-upscaled models, and vision-language models.
* The base models perform well on a range of benchmarks like MMLU, and the finetuned chat models have high human preference rates on evaluation platforms like AlpacaEval and Chatbot Arena.
* The performance of Yi models is primarily attributed to data quality resulting from data-engineering efforts.
* For pretraining, a cascaded data deduplication and quality filtering pipeline is used to construct 3.1 trillion tokens of English and Chinese corpora.
* For finetuning, a small scale (less than 10K) instruction dataset is polished over multiple iterations, with every instance verified directly by machine learning engineers.
* For vision-language, a vision transformer encoder is combined with the chat language model, and the model is trained to align visual representations to the semantic space of the language model.
* The context length is extended to 200K through lightweight continual pretraining, demonstrating strong needle-in-a-haystack retrieval performance.
* Extending the depth of the pretrained checkpoint through continual pretraining further improves performance.
* The authors believe that continuing to scale up model parameters with optimized data will lead to even stronger frontier models.

## Generative Representational Instruction Tuning

![pdf](papers/muennighoff2024generative/2402.09906.pdf)

 * The paper proposes a new method called Generative Representational Instruction Tuning (GRIT) that enables a large language model to handle both generative and embedding tasks by distinguishing between them through instructions.
* The resulting GRITLM 7B model sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks.
* By scaling up further, GRITLM 8X7B outperforms all open generative language models that were tried while still being among the best embedding models.
* GRIT matches training on only generative or embedding data, thus unifying both at no performance loss.
* The unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models.
* The models, code, and other materials are freely available at <https://github.com/ContextualAI/gritlm>.
* The paper includes a figure that compares the performance of various models on text representation (embedding) and generation tasks, showing that GRITLM is the first model to perform best-in-class at both types of tasks simultaneously.
* The paper is a preprint and is currently under review.
* The paper's arXiv ID is 2402.09906v1 [cs.CL], and it was published on 15 Feb 2024.

## Fast Model Editing at Scale

![pdf](papers/mitchell2022fast/2110.11309.pdf)

 * The paper proposes a method for editing large pre-trained models to correct inaccurate outputs or update outdated information.
* The proposed method, called Model Editor Networks with Gradient Decomposition (MEND), uses a single desired input-output pair to make fast, local edits to a pre-trained model's behavior.
* MEND learns to transform the gradient obtained by standard fine-tuning using a low-rank decomposition of the gradient to make the parameterization of this transformation tractable.
* MEND can be trained on a single GPU in less than a day even for 10 billion+ parameter models and enables rapid application of new edits to the pre-trained model.
* The authors compare MEND to other editing algorithms and find that it is the only approach that effectively edits the behavior of models with more than 10 billion parameters.
* The authors test MEND on T5, GPT, BERT, and BART models and find that it is able to make targeted edits while preserving the model's performance on unrelated inputs.
* The code and data for MEND are available at <https://sites.google.com/view/mend-editing>.

## GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

![pdf](papers/zhao2024galore/2403.03507.pdf)

 * Training Large Language Models (LLMs) is memory-intensive due to the growing size of weights and optimizer states.
* Common memory-reduction approaches like low-rank adaptation (LoRA) add a trainable low-rank matrix to the frozen pre-trained weight in each layer, reducing trainable parameters and optimizer states.
* However, LoRA and similar methods typically underperform training with full-rank weights in both pre-training and fine-tuning stages.
* The authors propose Gradient Low-Rank Projection (GaLore), a memory-efficient training strategy that allows full-parameter learning while reducing memory usage.
* GaLore reduces memory usage by up to 65.5% in optimizer states and maintains performance for pre-training on LLaMA 1B and 7B architectures with the C4 dataset.
* The 8-bit GaLore further reduces optimizer memory by up to 82.5% and total training memory by 63.3%, compared to a BF16 baseline.
* The authors demonstrate the feasibility of pre-training a 7B model on consumer GPUs with 24GB memory without model parallel, checkpointing, or offloading strategies.
* GaLore is a PyTorch-like algorithm that projects gradients to a compact space, updates them, and then projects them back to the original space.
* The authors show that GaLore outperforms other memory-reduction techniques like gradient checkpointing and activation offloading.
* GaLore is a promising approach for memory-efficient LLM training, allowing for full-parameter learning and reducing memory usage.

## sDPO: Don't Use Your Data All at Once

![pdf](papers/kim2024sdpo/2403.19270.pdf)

 * The paper proposes sDPO, an extension of Direct Preference Optimization (DPO) for aligning large language models (LLMs) with human preferences.
* DPO involves curating preference datasets using human or strong AI judgement to select chosen and rejected responses to questions, and training LLMs by comparing log probabilities of chosen versus rejected answers.
* The authors argue that the reference model used in DPO, which acts as a lower bound, is usually set as the base SFT model, which is a weaker alternative with potentially misaligned preferences.
* The proposed sDPO trains the final model to be more performant by using the aligned model in the previous step as the reference model for the current step, resulting in a more aligned reference model.
* The authors demonstrate that sDPO facilitates the use of more precisely aligned reference models within the DPO training framework.
* The sDPO trained models outperform other popular LLMs with more parameters, as shown in Table 1 with H4 scores for Mistral-7B-OpenOrca and OpenHermes-2.5-Mistral-7B with different reference models.
* The sDPO trained models perform better than the base SFT model, even when the base SFT model is already aligned, indicating that sDPO can improve the alignment of already aligned models.
* The authors also show that sDPO can be used with different types of reference models, including open source models and models trained with different techniques.
* The paper concludes that sDPO is a simple and effective method for aligning LLMs with human preferences, and can be used with different types of reference models.

## Gecko: Versatile Text Embeddings Distilled from Large Language Models

![pdf](papers/lee2024gecko/2403.20327.pdf)

 * Gecko is a new text embedding model that is compact and versatile, achieving strong retrieval performance.
* It distills knowledge from large language models (LLMs) into a retriever using a two-step distillation process.
* The first step generates diverse, synthetic paired data using an LLM, and the second step refines the data quality by retrieving a set of candidate passages for each query and relabeling the positive and hard negative passages using the same LLM.
* Gecko outperforms existing entries with 768 embedding size on the Massive Text Embedding Benchmark (MTEB) with 256 embedding dimensions.
* Gecko with 768 embedding dimensions competes with 7x larger models and 5x higher dimensional embeddings.
* Gecko's approach leverages insights from knowledge distillation to create a two-step LLM-powered embedding model.
* It starts with a large corpus of (unlabeled) passages, uses a few-shot prompted LLM to generate a relevant task and query for each passage, and embeds the concatenated task and query using a pretrained embedding model.
* It then reranks the passages using an LLM to obtain positive and negative passages based on the LLM scores.
* The reranking step is key to enhance the quality as the best passage to answer the generated query often differs from the original source passage.
* Using Gecko's LLM-based dataset, FRet, alone can lead to significant improvement, setting a strong baseline as a zero-shot embedding model on MTEB.

## Advancing LLM Reasoning Generalists with Preference Trees

![pdf](papers/yuan2024advancing/2404.02078.pdf)

 * The paper introduces EURUS, a suite of large language models (LLMs) optimized for reasoning.
* EURUS models are finetuned from Mistral-7B and CodeLlama-70B and achieve state-of-the-art results on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems.
* EURUS-70B beats GPT-3.5 Turbo in reasoning across 12 tests covering five tasks and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA.
* The strong performance of EURUS can be primarily attributed to ULTRAINTERACT, a newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks.
* ULTRAINTERACT includes a preference tree consisting of reasoning chains, multi-turn interaction trajectories, and pairwise data to facilitate preference learning.
* The investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks, and a novel reward modeling objective is derived.
* The strong reward model, together with ULTRAINTERACT, leads to a significant improvement in the performance of EURUS on reasoning tasks.
* The authors also release the EURUS models and ULTRAINTERACT dataset for research purposes.

## Simple and Scalable Strategies to Continually Pre-train Large Language Models

![pdf](papers/ibrahim2024simple/2403.08763.pdf)

 * Large language models (LLMs) are typically pre-trained on billions of tokens and then re-trained from scratch when new data becomes available.
* Continual pre-training of LLMs can save significant compute compared to re-training, but it can result in degraded performance on previous data or poor adaptation to new data due to distribution shift.
* The authors propose a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data to match the performance of fully re-training from scratch on all available data.
* The proposed method is shown to match the re-training baseline for a 405M parameter model on two commonly used LLM pre-training datasets (English→English) and a stronger distribution shift (English→German).
* The method also matches the re-training baseline for a 10B parameter LLM on a weak but realistic distribution shift (English→English).
* The authors propose alternatives to the cosine learning rate schedule to help circumvent forgetting induced by LR re-warming and that are not bound to a fixed token budget.

## ReFT: Representation Finetuning for Language Models

![pdf](papers/wu2024reft/2404.03592.pdf)

 * Representation Finetuning (ReFT) is a new approach to adapting pretrained language models (LMs) to new tasks or domains, which involves learning task-specific interventions on hidden representations rather than updating model weights.
* The authors propose a specific instance of the ReFT family, called Low-rank Linear Subspace ReFT (LoReFT), which is a drop-in replacement for existing parameter-efficient finetuning (PEFT) methods and learns interventions that are more parameter-efficient than prior state-of-the-art PEFTs.
* LoReFT is evaluated on eight commonsense reasoning tasks, four arithmetic reasoning tasks, Alpaca-Eval v1.0, and GLUE, and is shown to deliver the best balance of efficiency and performance, outperforming state-of-the-art PEFTs in most cases.
* The authors release a generic ReFT training library publicly at <https://github.com/stanfordnlp/pyreft>.
* Current state-of-the-art PEFTs modify model weights, but interpretability work has shown that representations encode rich semantic information, suggesting that editing representations might be a more powerful alternative to weight updates.
* ReFT methods operate on a frozen base model and learn task-specific interventions on hidden representations, reducing memory usage and training time while maintaining similar performance to full finetuning in many practical settings.
* Adapters, a common family of PEFTs, learn an edit that can be added to a subset of model weights or an additional set of weights that operate alongside the frozen base model.
* Recent adapters such as LoRA and DoRA use low-rank approximations in place of full weight matrices during adapter training, reducing the number of trainable parameters in learned weight updates.
* QLoRA further shows that full-precision adapters can be trained on top of reduced-precision models without sacrificing performance.
* Adapters are generally more efficient and effective than methods that introduce new model components, like prefix-tuning.

## RAFT: Adapting Language Model to Domain Specific RAG

![pdf](papers/zhang2024raft/2403.10131.pdf)

 * Pretrained Large Language Models (LLMs) are commonly adapted to specific domains through finetuning or RAG-based prompting.
* The optimal methodology for LLMs to gain new knowledge remains an open question.
* The authors propose Retrieval Augmented Fine Tuning (RAFT), a training recipe that improves the model's ability to answer questions in an "open-book" in-domain setting.
* RAFT trains the model to ignore irrelevant documents and cite verbatim the right sequence from the relevant document to answer the question.
* RAFT consistently improves the model's performance in domain-specific RAG, outperforming existing methods in PubMed, HotpotQA, and Gorilla datasets.
* RAFT's code and demo are open-sourced at <https://github.com/ShishirPatil/gorilla>.

## Reinforced Self-Training (ReST) for Language Modeling

![pdf](papers/gulcehre2023reinforced/2308.08998.pdf)

 * The paper proposes Reinforced Self-Training (ReST), a simple algorithm for aligning large language models (LLMs) with human preferences using reinforcement learning from human feedback (RLHF).
* ReST is inspired by growing batch reinforcement learning and is more efficient than typical online RLHF methods because it produces the training dataset offline, allowing data reuse.
* The ReST method consists of two steps: Grow and Improve. During the Grow step, a policy generates a dataset, and at the Improve step, the filtered dataset is used to fine-tune the policy.
* The authors focus on the application of ReST to machine translation and show that it can substantially improve translation quality, as measured by automated metrics and human evaluation.
* ReST is a general approach applicable to all generative learning settings, and the authors believe it can be used to improve the quality of LLMs' outputs in various tasks.
* The paper highlights the importance of aligning LLMs with human preferences to avoid generating unsafe or harmful contents and to improve performance on downstream tasks.
* The authors also discuss the limitations of ReST, such as the need for a high-quality initial policy and the potential for the reward model to overfit to the human feedback.
* The paper includes experiments and results that demonstrate the effectiveness of ReST in improving the translation quality of LLMs in a compute and sample-efficient manner.
* The authors suggest that ReST can be further improved by incorporating active learning techniques and by exploring different ways of generating the initial policy.
* The paper is a contribution of researchers from Google DeepMind and Google Research.

## OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset

![pdf](papers/toshniwal2024openmathinstruct1/2402.10176.pdf)

 * The paper introduces OpenMathInstruct-1, a dataset of 1.8 million math problem-solution pairs generated using the open-source Mixtral model.
* The dataset is used to train a model called OpenMath-CodeLlama-70B, which achieves competitive scores on the GSM8K and MATH benchmarks.
* The authors argue that the use of open-source models for generating math instruction tuning datasets has been limited due to the wide gap in mathematical skills between closed-source and open-source LLMs.
* The paper aims to address this gap by using a novel prompting strategy and scaling techniques to generate high-quality synthetic data.
* The authors compare the performance of Mixtral with GPT-4, currently one of the best closed-source LLMs for mathematical reasoning, and find that Mixtral performs well but still lags behind GPT-4.
* The OpenMathInstruct-1 dataset, code, and models are released under a commercially permissive license.
* The paper highlights the limitations of using proprietary models like GPT-4 for model development, including legal restraints, cost, and lack of reproducibility.
* The authors argue that open-source models like Mixtral can provide a viable alternative for developing mathematical reasoning models, with the potential to overcome these limitations.

## Instruction-tuned Language Models are Better Knowledge Learners

![pdf](papers/jiang2024instructiontuned/2402.12847.pdf)

 * Large language models (LLMs) store factual knowledge in their parameters through pre-training, but this knowledge can become outdated or insufficient over time.
* Continued pre-training on new documents can help keep LLMs up-to-date, and instruction-tuning can make it easier to elicit this knowledge.
* However, even with minimized perplexity, the amount of elicited knowledge is still limited, a phenomenon referred to as the "perplexity curse."
* The authors propose pre-instruction-tuning (PIT), a method that instruction-tunes on questions before training on documents, to mitigate the perplexity curse.
* PIT significantly enhances the ability of LLMs to absorb knowledge from new documents, outperforming standard instruction-tuning by 17.8%.

## Recurrent Drafter for Fast Speculative Decoding in Large Language Models

![pdf](papers/zhang2024recurrent/2403.09919.pdf)

 * The paper proposes an improved approach for speculative decoding to enhance the efficiency of serving large language models (LLMs).
* The approach combines the strengths of the classic two-model speculative decoding and the single-model approach, Medusa.
* It uses a single-model strategy with a lightweight draft head that has a recurrent dependency design, similar to the small draft model in classic speculative decoding.
* The recurrent dependency allows for swift filtering of undesired candidates using beam search, avoiding the need for a data-dependent tree attention structure during inference as in Medusa.
* The proposed method is empirically demonstrated to be effective on several popular open-source LLMs, with a comprehensive analysis of trade-offs involved.
* LLMs are large models with billions of parameters, using auto-regressive methods for token-by-token responses, which can be slow due to memory bandwidth constraints and large model size.
* Speculative decoding has emerged as a promising strategy to accelerate LLM inference, using a smaller draft model to generate preliminary candidate tokens and a larger target model for verification.
* The single-model approach, as in Medusa, is preferred for easier integration into existing LLM serving systems.
* The proposed method simplifies the single-model design while avoiding the complexities of the full transformer architecture and the need for a data-dependent tree attention structure.

## Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM

![pdf](papers/sukhbaatar2024branchtrainmix/2403.07816.pdf)

 * The paper proposes Branch-Train-MiX (BTX), a method for training Large Language Models (LLMs) that can specialize in multiple domains such as coding, math, and world knowledge.
* BTX starts by branching a seed model to train experts in parallel with high throughput and reduced communication cost.
* After individual experts are trained, BTX combines their feedforward parameters as experts in Mixture-of-Expert (MoE) layers and averages the remaining parameters.
* BTX then finetunes the MoE layers to learn token-level routing, which is not present in the Branch-Train-Merge method.
* BTX generalizes two special cases, the Branch-Train-Merge method and sparse upcycling, which omits the stage of training experts asynchronously.
* BTX achieves the best accuracy-efficiency tradeoff compared to alternative approaches.
* Recent work has proposed the Branch-Train-Merge (BTM) method for embarrassingly parallel training of LLMs without synchronization, but its main drawback is the lack of a unified single model for further finetuning.
* The Mixture-of-Experts (MoE) approach reduces the computational footprint of LLMs by keeping only a subset of parameters active at any given time, allowing the total number of parameters to grow without additional computation.
* MoE has shown impressive performance on downstream tasks, but it is often trained in a fully synchronized manner.
* BTX combines the benefits of both BTM and MoE, allowing for efficient and effective training of LLMs that can specialize in multiple domains while still being finetuned for specific tasks.

## InternLM2 Technical Report

![pdf](papers/cai2024internlm2/2403.17297.pdf)

 * The paper introduces InternLM2, an open-source Large Language Model (LLM) that outperforms previous models in comprehensive evaluations across 6 dimensions and 30 benchmarks.
* InternLM2 is trained on diverse data types, including text, code, and long-context data, and efficiently captures long-term dependencies.
* The model is initially trained on 4k tokens before advancing to 32k tokens in pre-training and fine-tuning stages, exhibiting remarkable performance on the 200k "Needle-in-a-Haystack" test.
* InternLM2 is aligned using Supervised Fine-Tuning (SFT) and a novel Conditional Online Reinforcement Learning from Human Feedback (COOL RLHF) strategy that addresses conflicting human preferences and reward hacking.
* The authors release InternLM2 models in different training stages and model sizes to provide the community with insights into the model's evolution.
* The paper aims to replicate the advancements of closed-source LLMs like ChatGPT and GPT-4 in open-source models.
* The pre-training process of InternLM2 is detailed, highlighting the preparation of diverse data types and the use of innovative pre-training and optimization techniques.
* The model is evaluated on long-context modeling and open-ended subjective evaluations, showing superior performance compared to previous models.
* The authors address the challenges of replicating advancements in open-source models and aim to provide a more transparent and accessible approach to LLM development.

