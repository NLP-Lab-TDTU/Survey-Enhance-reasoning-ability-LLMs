# References

## Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Authors: Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou

[Link to PDF](papers/wei2023chainofthought/2201.11903.pdf)

 * The paper proposes a method called "chain-of-thought" prompting to improve the complex reasoning abilities of large language models.
* This method involves providing a few chain of thought demonstrations as exemplars in prompting, which leads to the emergence of reasoning abilities in the models.
* The authors conduct experiments on three large language models and show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks.
* The empirical gains from this method can be significant, as demonstrated by a PaLM 540B model achieving state-of-the-art accuracy on the GSM8K benchmark of math word problems.
* The paper also provides examples of the chain-of-thought reasoning process, highlighting how the models break down complex problems into intermediate reasoning steps.
* The authors suggest that this method could be useful for a wide range of applications, including decision-making, tutoring, and explanation generation.

Citation:
```
@misc{wei2023chainofthought,
      title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
      author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
      year={2023},
      eprint={2201.11903},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}X

```

## Yi: Open Foundation Models by 01.AI

Authors: 01. AI and : and Alex Young and Bei Chen and Chao Li and Chengen Huang and Ge Zhang and Guanwei Zhang and Heng Li and Jiangcheng Zhu and Jianqun Chen and Jing Chang and Kaidong Yu and Peng Liu and Qiang Liu and Shawn Yue and Senbin Yang and Shiming Yang and Tao Yu and Wen Xie and Wenhao Huang and Xiaohui Hu and Xiaoyi Ren and Xinyao Niu and Pengcheng Nie and Yuchi Xu and Yudong Liu and Yue Wang and Yuxuan Cai and Zhenyu Gu and Zhiyuan Liu and Zonghong Dai

[Link to PDF](papers/ai2024yi/2403.04652.pdf)

 * The Yi model family is a series of language and multimodal models with strong multi-dimensional capabilities.
* The models are based on 6B and 34B pretrained language models, and are extended to chat models, 200K long context models, depth-upscaled models, and vision-language models.
* The base models perform well on a range of benchmarks, and the finetuned chat models have a high human preference rate on evaluation platforms.
* The performance of the Yi models is attributed to data quality resulting from data-engineering efforts, including a cascaded data deduplication and quality filtering pipeline for pretraining and manual verification of a small-scale instruction dataset for finetuning.
* The vision-language models combine a chat language model with a vision transformer encoder and are trained to align visual representations to the semantic space of the language model.
* The context length is extended to 200K through lightweight continual pretraining, and the performance is further improved by extending the depth of the pretrained checkpoint through continual pretraining.

Citation:
```
@misc{ai2024yi,
      title={Yi: Open Foundation Models by 01.AI},
      author={01. AI and : and Alex Young and Bei Chen and Chao Li and Chengen Huang and Ge Zhang and Guanwei Zhang and Heng Li and Jiangcheng Zhu and Jianqun Chen and Jing Chang and Kaidong Yu and Peng Liu and Qiang Liu and Shawn Yue and Senbin Yang and Shiming Yang and Tao Yu and Wen Xie and Wenhao Huang and Xiaohui Hu and Xiaoyi Ren and Xinyao Niu and Pengcheng Nie and Yuchi Xu and Yudong Liu and Yue Wang and Yuxuan Cai and Zhenyu Gu and Zhiyuan Liu and Zonghong Dai},
      year={2024},
      eprint={2403.04652},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Generative Representational Instruction Tuning

Authors: Niklas Muennighoff and Hongjin Su and Liang Wang and Nan Yang and Furu Wei and Tao Yu and Amanpreet Singh and Douwe Kiela

[Link to PDF](papers/muennighoff2024generative/2402.09906.pdf)

 * The paper introduces a new method called Generative Representational Instruction Tuning (GRIT) that enables a large language model to handle both generative and embedding tasks by distinguishing between them through instructions.
* The resulting GRITLM 7B model sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks.
* GRITLM 8X7B outperforms all open generative language models while still being among the best embedding models.
* GRIT matches training on only generative or embedding data, thus unifying both at no performance loss.
* The unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents.
* The models, code, and other materials are available at <https://github.com/ContextualAI/gritlm>.

Citation:
```
@misc{muennighoff2024generative,
      title={Generative Representational Instruction Tuning},
      author={Niklas Muennighoff and Hongjin Su and Liang Wang and Nan Yang and Furu Wei and Tao Yu and Amanpreet Singh and Douwe Kiela},
      year={2024},
      eprint={2402.09906},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Fast Model Editing at Scale

Authors: Eric Mitchell and Charles Lin and Antoine Bosselut and Chelsea Finn and Christopher D. Manning

[Link to PDF](papers/mitchell2022fast/2110.11309.pdf)

 * The paper proposes a method for editing large pre-trained models to correct inaccurate outputs or update outdated information.
* The method, called Model Editor Networks with Gradient Decomposition (MEND), uses a collection of small auxiliary editing networks to make fast, local edits to a pre-trained model's behavior.
* MEND learns to transform the gradient obtained by standard fine-tuning using a low-rank decomposition of the gradient to make the parameterization of this transformation tractable.
* MEND can be trained on a single GPU in less than a day even for 10 billion+ parameter models and enables rapid application of new edits to the pre-trained model.
* The experiments on T5, GPT, BERT, and BART models show that MEND is the only approach to model editing that effectively edits the behavior of models with more than 10 billion parameters.
* The code and data are available at <https://sites.google.com/view/mend-editing>.

Citation:
```
@misc{mitchell2022fast,
      title={Fast Model Editing at Scale},
      author={Eric Mitchell and Charles Lin and Antoine Bosselut and Chelsea Finn and Christopher D. Manning},
      year={2022},
      eprint={2110.11309},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

Authors: Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei

[Link to PDF](papers/ma2024era/2402.17764.pdf)

 * The paper introduces a 1-bit Large Language Model (LLM) variant, BitNet b1.58, where every parameter is ternary {-1, 0, 1}.
* BitNet b1.5

Citation:
```
@misc{ma2024era,
      title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
      author={Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
      year={2024},
      eprint={2402.17764},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

Authors: Jiawei Zhao and Zhenyu Zhang and Beidi Chen and Zhangyang Wang and Anima Anandkumar and Yuandong Tian

[Link to PDF](papers/zhao2024galore/2403.03507.pdf)

 * Training Large Language Models (LLMs) is memory-intensive, with optimizer states and gradients taking up more memory than the trainable parameters themselves.
* Current memory-reduction methods, such as low-rank adaptation (LoRA), add a trainable low-rank matrix to the frozen pre-trained weight in each layer, reducing trainable parameters and optimizer states.
* However, these methods typically underperform training with full-rank weights in both pre-training and fine-tuning stages, as they limit the parameter search to a low-rank subspace and alter the training dynamics.
* The authors propose Gradient Low-Rank Projection (GaLore), a memory-efficient training strategy that allows full-parameter learning while reducing memory usage by up to 65.5% in optimizer states.
* GaLore maintains both efficiency and performance for pre-training on LLaMA 1B and 7B architectures with the C4 dataset and fine-tuning RoBERTa on GLUE tasks.
* The authors demonstrate, for the first time, the feasibility of pre-training a 7B model on consumer GPUs with 24GB memory without model parallel, checkpointing, or offloading strategies.

Citation:
```
@misc{zhao2024galore,
      title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection},
      author={Jiawei Zhao and Zhenyu Zhang and Beidi Chen and Zhangyang Wang and Anima Anandkumar and Yuandong Tian},
      year={2024},
      eprint={2403.03507},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## sDPO: Don't Use Your Data All at Once

Authors: Dahyun Kim and Yungi Kim and Wonho Song and Hyeonwoo Kim and Yunsu Kim and Sanghoon Kim and Chanjun Park

[Link to PDF](papers/kim2024sdpo/2403.19270.pdf)

 * The paper proposes sDPO, an extension of Direct Preference Optimization (DPO) for aligning large language models (LLMs) with human preferences.
* sDPO involves dividing available preference datasets and utilizing them in a step-wise manner during DPO training, allowing for more precise alignment with reference models.
* The authors demonstrate that sDPO results in a final model that is more performant than other LLMs with more parameters.
* DPO involves curating preference datasets using human or strong AI judgement to select chosen and rejected responses to questions, and training LLMs by comparing log probabilities of chosen versus rejected answers.
* However, obtaining these probabilities can be challenging with proprietary models, and the reference model is typically set as the base SFT model, which is a weaker alternative with potentially misaligned preferences.
* sDPO addresses this issue by using the aligned model from the previous step as the reference model for the current step, resulting in a more aligned reference model and a more performant final model.

Citation:
```
@misc{kim2024sdpo,
      title={sDPO: Don't Use Your Data All at Once}, 
      author={Dahyun Kim and Yungi Kim and Wonho Song and Hyeonwoo Kim and Yunsu Kim and Sanghoon Kim and Chanjun Park},
      year={2024},
      eprint={2403.19270},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## Gecko: Versatile Text Embeddings Distilled from Large Language Models

Authors: Jinhyuk Lee and Zhuyun Dai and Xiaoqi Ren and Blair Chen and Daniel Cer and Jeremy R. Cole and Kai Hui and Michael Boratko and Rajvi Kapadia and Wen Ding and Yi Luan and Sai Meher Karthik Duddu and Gustavo Hernandez Abrego and Weiqiang Shi and Nithi Gupta and Aditya Kusupati and Prateek Jain and Siddhartha Reddy Jonnalagadda and Ming-Wei Chang and Iftekhar Naim

[Link to PDF](papers/lee2024gecko/2403.20327.pdf)

 * Gecko is a new text embedding model that is compact and versatile, achieving strong retrieval performance.
* It distills knowledge from large language models (LLMs) into a retriever using a two-step process.
* The first step generates diverse, synthetic paired data using an LLM, and the second step refines the data quality by retrieving a set of candidate passages for each query and relabeling the positive and hard negative passages using the same LLM.
* Gecko outperforms existing entries with 768 embedding size and competes with 7x larger models and 5x higher dimensional embeddings.
* The authors show that using their LLM-based dataset, FRet, alone can lead to significant improvement, setting a strong baseline as a zero-shot embedding model on MTEB.

Citation:
```
@misc{lee2024gecko,
      title={Gecko: Versatile Text Embeddings Distilled from Large Language Models},
      author={Jinhyuk Lee and Zhuyun Dai and Xiaoqi Ren and Blair Chen and Daniel Cer and Jeremy R. Cole and Kai Hui and Michael Boratko and Rajvi Kapadia and Wen Ding and Yi Luan and Sai Meher Karthik Duddu and Gustavo Hernandez Abrego and Weiqiang Shi and Nithi Gupta and Aditya Kusupati and Prateek Jain and Siddhartha Reddy Jonnalagadda and Ming-Wei Chang and Iftekhar Naim},
      year={2024},
      eprint={2403.20327},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Direct Preference Optimization: Your Language Model is Secretly a Reward Model

Authors: Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn

[Link to PDF](papers/rafailov2023direct/2305.18290.pdf)

 * Large language models (LMs) trained on extensive datasets exhibit impressive capabilities but may also replicate undesirable behaviors from their human trainers.
* Current methods for aligning LMs with human preferences typically involve reinforcement learning (RL) to fine-tune the models, but these methods can be complex, unstable, and computationally expensive.
* The authors propose Direct Preference Optimization (DPO), a new parameterization of the reward model in RLHF that allows for closed-form extraction of the optimal policy, simplifying the process to a simple classification loss.
* DPO is stable, efficient, and eliminates the need for sampling from the LM during fine-tuning or hyperparameter tuning.
* Experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods, improving control over sentiment in generations and matching or improving response quality in summarization and single-turn dialogue.

Citation:
```
@misc{rafailov2023direct,
      title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
      author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.18290},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Advancing LLM Reasoning Generalists with Preference Trees

Authors: Lifan Yuan and Ganqu Cui and Hanbin Wang and Ning Ding and Xingyao Wang and Jia Deng and Boji Shan and Huimin Chen and Ruobing Xie and Yankai Lin and Zhenghao Liu and Bowen Zhou and Hao Peng and Zhiyuan Liu and Maosong Sun

[Link to PDF](papers/yuan2024advancing/2404.02078.pdf)

 * The paper introduces EURUS, a suite of large language models (LLMs) optimized for reasoning, which achieve state-of-the-art results on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems.
* EURUS-70B beats GPT-3.5 Turbo in reasoning across five tasks and substantially outperforms existing open-source models on LeetCode and TheoremQA by margins more than 13.3%.
* The strong performance of EURUS can be primarily attributed to ULTRAINTERACT, a newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks.
* ULTRAINTERACT includes a preference tree consisting of reasoning chains, multi-turn interaction trajectories, and pairwise data to facilitate preference learning.
* The investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks, and a novel reward modeling objective is derived, leading to a strong reward model.

Citation:
```
@misc{yuan2024advancing,
      title={Advancing LLM Reasoning Generalists with Preference Trees}, 
      author={Lifan Yuan and Ganqu Cui and Hanbin Wang and Ning Ding and Xingyao Wang and Jia Deng and Boji Shan and Huimin Chen and Ruobing Xie and Yankai Lin and Zhenghao Liu and Bowen Zhou and Hao Peng and Zhiyuan Liu and Maosong Sun},
      year={2024},
      eprint={2404.02078},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

```

## BitNet: Scaling 1-bit Transformers for Large Language Models

Authors: Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei

[Link to PDF](papers/wang2023bitnet/2310.11453.pdf)

 * The paper introduces BitNet, a scalable and stable 1-bit Transformer architecture for large language models.
* BitNet uses a drop-in replacement called BitLinear for the nn.Linear layer to train 1-bit weights from scratch.
* Experimental results show that BitNet achieves competitive performance while reducing memory footprint and energy consumption, compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines.
* BitNet exhibits a scaling law similar to full-precision Transformers, suggesting its potential for effective scaling to larger language models while maintaining efficiency and performance benefits.
* BitNet trains 1-bit Transformers from scratch, significantly outperforming state-of-the-art quantization methods and achieving energy-efficient results.
* The cost savings of BitNet become more significant as the model size scales up, while still achieving competitive performance with models trained with FP16.

Citation:
```
@misc{wang2023bitnet,
      title={BitNet: Scaling 1-bit Transformers for Large Language Models},
      author={Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
      year={2023},
      eprint={2310.11453},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Simple and Scalable Strategies to Continually Pre-train Large Language Models

Authors: Adam Ibrahim and Benjamin Thérien and Kshitij Gupta and Mats L. Richter and Quentin Anthony and Timothée Lesort and Eugene Belilovsky and Irina Rish

[Link to PDF](papers/ibrahim2024simple/2403.08763.pdf)

 * Large language models (LLMs) are typically pre-trained on billions of tokens, and the process is repeated when new data becomes available.
* Continual pre-training of LLMs is more efficient, but the distribution shift in new data can result in degraded performance on previous data or poor adaptation to new data.
* The authors propose a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data to match the performance of fully re-training from scratch.
* The proposed method is shown to match the performance of re-training from scratch on two commonly used LLM pre-training datasets (English→English) and a stronger distribution shift (English→German) at the 405M parameter model scale.
* The method also matches the re-training baseline for a 10B parameter LLM, demonstrating that LLMs can be successfully updated via simple and scalable continual learning strategies, using only a fraction of the compute.
* The authors also propose alternatives to the cosine learning rate schedule to help circumvent forgetting induced by LR re-warming and that are not bound to a fixed token budget.

Citation:
```
@misc{ibrahim2024simple,
      title={Simple and Scalable Strategies to Continually Pre-train Large Language Models},
      author={Adam Ibrahim and Benjamin Thérien and Kshitij Gupta and Mats L. Richter and Quentin Anthony and Timothée Lesort and Eugene Belilovsky and Irina Rish},
      year={2024},
      eprint={2403.08763},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## ReFT: Representation Finetuning for Language Models

Authors: Zhengxuan Wu and Aryaman Arora and Zheng Wang and Atticus Geiger and Dan Jurafsky and Christopher D. Manning and Christopher Potts

[Link to PDF](papers/wu2024reft/2404.03592.pdf)

 * Representation Finetuning (ReFT) is a new approach to adapting pretrained language models to new tasks, which involves learning task-specific interventions on hidden representations instead of updating model weights.
* The authors propose a specific instance of ReFT called Low-rank Linear Subspace ReFT (LoReFT), which is a drop-in replacement for existing parameter-efficient finetuning (PEFT) methods and learns interventions that are more parameter-efficient than prior state-of-the-art PEFTs.
* LoReFT is evaluated on eight commonsense reasoning tasks, four arithmetic reasoning tasks, Alpaca-Eval v1.0, and GLUE, and is shown to deliver the best balance of efficiency and performance, outperforming state-of-the-art PEFTs in most cases.
* The authors release a generic ReFT training library publicly at <https://github.com/stanfordnlp/pyreft>.
* ReFT methods offer a promising alternative to traditional PEFT methods, as they can leverage the rich semantic information encoded in representations to achieve better performance with fewer trainable parameters.

Citation:
```
@misc{wu2024reft,
      title={ReFT: Representation Finetuning for Language Models}, 
      author={Zhengxuan Wu and Aryaman Arora and Zheng Wang and Atticus Geiger and Dan Jurafsky and Christopher D. Manning and Christopher Potts},
      year={2024},
      eprint={2404.03592},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## RAFT: Adapting Language Model to Domain Specific RAG

Authors: Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez

[Link to PDF](papers/zhang2024raft/2403.10131.pdf)

 * Pretrained Large Language Models (LLMs) are commonly adapted to specific domains through RAG-based prompting or finetuning.
* The optimal methodology for LLMs to gain new knowledge remains an open question.
* The authors propose Retrieval Augmented Fine Tuning (RAFT), a training recipe that improves the model's ability to answer questions in an "open-book" in-domain setting.
* RAFT trains the model to ignore irrelevant documents and cite verbatim the right sequence from the relevant document to answer the question.
* RAFT consistently improves the model's performance in domain-specific RAG, outperforming existing methods in PubMed, HotpotQA, and Gorilla datasets.
* The authors open-source the code and demo for RAFT at <https://github.com/ShishirPatil/gorilla>.

Citation:
```
@misc{zhang2024raft,
      title={RAFT: Adapting Language Model to Domain Specific RAG},
      author={Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez},
      year={2024},
      eprint={2403.10131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Reinforced Self-Training (ReST) for Language Modeling

Authors: Caglar Gulcehre and Tom Le Paine and Srivatsan Srinivasan and Ksenia Konyushkova and Lotte Weerts and Abhishek Sharma and Aditya Siddhant and Alex Ahern and Miaosen Wang and Chenjie Gu and Wolfgang Macherey and Arnaud Doucet and Orhan Firat and Nando de Freitas

[Link to PDF](papers/gulcehre2023reinforced/2308.08998.pdf)

 * The paper proposes Reinforced Self-Training (ReST), a simple algorithm for aligning large language models (LLMs) with human preferences using reinforcement learning from human feedback (RLHF).
* ReST is inspired by growing batch reinforcement learning and is more efficient than typical online RLHF methods because it produces the training dataset offline, allowing data reuse.
* The paper focuses on the application of ReST to machine translation and shows that it can substantially improve translation quality, as measured by automated metrics and human evaluation.
* ReST consists of two steps: Grow, where a policy generates a dataset, and Improve, where the filtered dataset is used to fine-tune the policy. These steps are repeated, with Improve step repeated more frequently to amortize the dataset creation cost.
* The authors demonstrate that ReST can align LLMs with human preferences, improving their performance on downstream tasks and reducing the risk of generating unsafe or harmful content.
* ReST is a general approach applicable to all generative learning settings, and the authors suggest that it could be used to improve the performance of LLMs in other domains.

Citation:
```
@misc{gulcehre2023reinforced,
      title={Reinforced Self-Training (ReST) for Language Modeling},
      author={Caglar Gulcehre and Tom Le Paine and Srivatsan Srinivasan and Ksenia Konyushkova and Lotte Weerts and Abhishek Sharma and Aditya Siddhant and Alex Ahern and Miaosen Wang and Chenjie Gu and Wolfgang Macherey and Arnaud Doucet and Orhan Firat and Nando de Freitas},
      year={2023},
      eprint={2308.08998},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset

Authors: Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman

[Link to PDF](papers/toshniwal2024openmathinstruct1/2402.10176.pdf)

 * The paper introduces OpenMathInstruct-1, a new math instruction tuning dataset with 1.8 million problem-solution pairs.
* The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH benchmarks using the Mixtral model.
* The authors compare the mathematical skills of the best closed-source LLMs, such as GPT-4, and the best open-source LLMs and find a wide gap.
* They propose a prompting novelty and scaling to construct the OpenMathInstruct-1 dataset, which is released under a commercially permissive license.
* The best model, OpenMath-CodeLlama-70B, trained on a subset of OpenMathInstruct-1, achieves a score of 84.6% on GSM8K and 50.7% on MATH, which is competitive with the best gpt-distilled models.
* The paper highlights the limitations of using proprietary models like GPT-4 for mathematical reasoning model development, such as legal restraints, cost, and lack of reproducibility.

Citation:
```
@misc{toshniwal2024openmathinstruct1,
      title={OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset},
      author={Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
      year={2024},
      eprint={2402.10176},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Instruction-tuned Language Models are Better Knowledge Learners

Authors: Zhengbao Jiang and Zhiqing Sun and Weijia Shi and Pedro Rodriguez and Chunting Zhou and Graham Neubig and Xi Victoria Lin and Wen-tau Yih and Srinivasan Iyer

[Link to PDF](papers/jiang2024instructiontuned/2402.12847.pdf)

 * Large language models (LLMs) store factual knowledge in their parameters through pre-training, but this knowledge can become outdated or insufficient over time.
* Continued pre-training on new documents can help keep LLMs up-to-date, and instruction-tuning can make it easier to elicit this knowledge.
* However, even with minimized perplexity, the amount of elicited knowledge is still limited, a phenomenon referred to as the "perplexity curse."
* The paper proposes pre-instruction-tuning (PIT), a method that instruction-tunes on questions prior to training on documents, to mitigate the perplexity curse.
* PIT significantly enhances the ability of LLMs to absorb knowledge from new documents, outperforming standard instruction-tuning by 17.8%.

Citation:
```
@misc{jiang2024instructiontuned,
      title={Instruction-tuned Language Models are Better Knowledge Learners},
      author={Zhengbao Jiang and Zhiqing Sun and Weijia Shi and Pedro Rodriguez and Chunting Zhou and Graham Neubig and Xi Victoria Lin and Wen-tau Yih and Srinivasan Iyer},
      year={2024},
      eprint={2402.12847},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Recurrent Drafter for Fast Speculative Decoding in Large Language Models

Authors: Aonan Zhang and Chong Wang and Yi Wang and Xuanyu Zhang and Yunfei Cheng

[Link to PDF](papers/zhang2024recurrent/2403.09919.pdf)

 * The paper proposes an improved approach for speculative decoding to enhance the efficiency of serving large language models (LLMs).
* The method combines the strengths of the classic two-model speculative decoding approach and the single-model approach, Medusa.
* It uses a single-model strategy with a lightweight draft head that has a recurrent dependency design, similar to the small draft model in classic speculative decoding but without the complexities of the full transformer architecture.
* The recurrent dependency allows for swift filtering of undesired candidates using beam search, resulting in a method that is simple in design and avoids creating a data-dependent tree attention structure only for inference, as in Medusa.
* The proposed method is empirically demonstrated to be effective on several popular open-source language models, with a comprehensive analysis of trade-offs involved in adoption.
* Large language models use auto-regressive methods to generate token-by-token responses, and the latency of the single token generation step significantly increases with model size.
* Speculative decoding has emerged as a promising strategy to accelerate LLM inference, using a smaller draft model to generate preliminary candidate tokens more efficiently, followed by verification by the larger target model.
* The single-model approach, as exemplified by Medusa, holds promise for easier integration into existing LLM serving systems.

Citation:
```
@misc{zhang2024recurrent,
      title={Recurrent Drafter for Fast Speculative Decoding in Large Language Models},
      author={Aonan Zhang and Chong Wang and Yi Wang and Xuanyu Zhang and Yunfei Cheng},
      year={2024},
      eprint={2403.09919},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM

Authors: Sainbayar Sukhbaatar and Olga Golovneva and Vasu Sharma and Hu Xu and Xi Victoria Lin and Baptiste Rozière and Jacob Kahn and Daniel Li and Wen-tau Yih and Jason Weston and Xian Li

[Link to PDF](papers/sukhbaatar2024branchtrainmix/2403.07816.pdf)

 * The paper proposes Branch-Train-MiX (BTX), a method for training Large Language Models (LLMs) that can specialize in multiple domains such as coding, math reasoning, and world knowledge.
* BTX starts by branching a seed model to train experts in parallel with high throughput and reduced communication cost.
* After individual experts are trained, BTX combines their feedforward parameters as experts in Mixture-of-Expert (MoE) layers and averages the remaining parameters, followed by an MoE-finetuning stage to learn token-level routing.
* BTX generalizes two special cases, the Branch-Train-Merge method and sparse upcycling.
* Compared to alternative approaches, BTX achieves the best accuracy-efficiency tradeoff.
* BTX allows for further supervised finetuning and reinforcement learning from human feedback, unlike the Branch-Train-Merge method.
* The Mixture-of-Experts approach is used to reduce the computational footprint of LLMs, allowing the total number of parameters to grow without additional computation.

Citation:
```
@misc{sukhbaatar2024branchtrainmix,
      title={Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM},
      author={Sainbayar Sukhbaatar and Olga Golovneva and Vasu Sharma and Hu Xu and Xi Victoria Lin and Baptiste Rozière and Jacob Kahn and Daniel Li and Wen-tau Yih and Jason Weston and Xian Li},
      year={2024},
      eprint={2403.07816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## InternLM2 Technical Report

Authors: Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin

[Link to PDF](papers/cai2024internlm2/2403.17297.pdf)

 * The paper introduces InternLM2, an open-source Large Language Model (LLM) that outperforms previous models in comprehensive evaluations across 6 dimensions and 30 benchmarks.
* InternLM2 is trained using innovative pre-training and optimization techniques, with a meticulous pre-training process that includes the preparation of diverse data types such as text, code, and long-context data.
* The model efficiently captures long-term dependencies, initially trained on 4k tokens before advancing to 32k tokens in pre-training and fine-tuning stages, exhibiting remarkable performance on the 200k "Needle-in-a-Haystack" test.
* InternLM2 is further aligned using Supervised Fine-Tuning (SFT) and a novel Conditional Online Reinforcement Learning from Human Feedback (COOL RLHF) strategy that addresses conflicting human preferences and reward hacking.
* The authors release InternLM2 models in different training stages and model sizes, providing the community with insights into the model’s evolution.

Citation:
```
@misc{cai2024internlm2,
      title={InternLM2 Technical Report},
      author={Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin},
      year={2024},
      eprint={2403.17297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

