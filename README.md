# A Survey of LLM × DATA

> A collection of papers and projects related to LLMs and corresponding data-centric methods. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://arxiv.org/abs/2505.18458)
>
> Other publicly-available materials: [[Slide]](./assets/DATA4LLM-en-v1.pdf)
>
> If you find our survey useful, please cite the paper:

```
@article{LLMDATASurvey,
    title={A Survey of LLM × DATA},
    author={Xuanhe Zhou, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, Fan Wu},
    year={2025},
    journal={arXiv preprint arXiv:2505.18458},
    url={https://arxiv.org/abs/2505.18458}
}
@article{tangllmasanalyst,
    title={LLM/Agent-as-Data-Analyst: A Survey},
    author={Zirui Tang, Weizheng Wang, Zihang Zhou, Yang Jiao, Bangrui Xu, Boyu Niu, Xuanhe Zhou, Guoliang Li, Yeye He, Wei Zhou, Yitong Song, Cheng Tan, Bin Wang, Conghui He, Xiaoyang Wang, Fan Wu},
    year={2025},
    journal={arXiv preprint arXiv:2509.23988},
    url={https://arxiv.org/abs/2509.23988}
}
```



## 🌤 The IaaS Concept of DATA4LLM



The **IaaS** concept for LLM data (phonetically echoing *Infrastructure as a Service*) defines the characteristics of high-quality datasets along four key dimensions: (1) **Inclusiveness** ensures broad coverage across domains, tasks, sources, languages, styles, and modalities. (2) **Abundance** emphasizes sufficient and well-balanced data volume to support scaling, fine-tuning, and continual learning without overfitting. (3) **Articulation** requires clear, coherent, and instructive content with step-by-step reasoning to enhance model understanding and task performance. (4) **Sanitization** involves rigorous filtering to remove private, toxic, unethical, and misleading content, ensuring data safety, neutrality, and compliance. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/data_llm_survey_v3.pdf)

[![Cover](https://github.com/SUPERZJ827/LLM4DB/raw/main/assets/iaas_overview_v2.png)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/iaas_overview_v2.png)

## 🌟 LLM/Agent-as-Data-Analyst

We observe the evolution of **LLM/Agent-as-Data-Analyst** techniques follows a five-dimension trajectory: (1) Data Modality (homogeneous → heterogeneous); (2) Analysis Functionality (literal → semantic); (3) Knowledge Scope (closed-world →open-world); (4) Tool Integration (tool-coupled → tool-assisted); (5) Development Autonomy (manual → fully autonomous).

[![Cover2](./assets/data_analysis_survey.png)](./assets/data_analysis_survey.png)


## Table of Contents

- [Datasets](#datasets)
- [0 Data Characteristics across LLM Stages](#0-data-characteristics-across-llm-stages)
  - [Data for Pretraining](#data-for-pretraining)
  - [Data for Continual Pre-training](#data-for-continual-pre-training)
  - [Data for Supervised Fine-Tuning (SFT)](#data-for-supervised-fine-tuning-sft)
  - [Data for Reinforcement Learning (RL)](#data-for-reinforcement-learning-rl)
  - [Data for Retrieval-Augmented Generation (RAG)](#data-for-retrieval-augmented-generation-rag)
  - [Data for LLM Evaluation](#data-for-llm-evaluation)
  - [Data for LLM Agents](#data-for-llm-agents)
- [1 Data Processing for LLM](#1-data-processing-for-llm)
  - [1.1 Data Acquisition](#11-data-acquisition)
  - [1.2 Data Deduplication](#12-data-deduplication)
  - [1.3 Data Filtering](#13-data-filtering)
  - [1.4 Data Selection](#14-data-selection)
  - [1.5 Data Mixing](#15-data-mixing)
  - [1.6 Data Distillation and Synthesis](#16-data-distillation-and-synthesis)
  - [1.7 End-to-End Data Processing Pipelines](#17-end-to-end-data-processing-pipelines)
- [2 Data Storage for LLM](#2-data-storage-for-llm)
  - [2.1 Data Formats](#21-data-formats)
  - [2.2 Data Distribution](#22-data-distribution)
  - [2.3 Data Organization](#23-data-organization)
  - [2.4 Data Movement](#24-data-movement)
  - [2.5 Data Fault Tolerance](#25-data-fault-tolerance)
  - [2.6 KV Cache](#26-kv-cache)
- [3 Data Serving for LLM](#3-data-serving-for-llm)
  - [3.1 Data Shuffling](#31-data-shuffling)
  - [3.2 Data Compression](#32-data-compression)
  - [3.3 Data Packing](#33-data-packing)
  - [3.4 Data Provenance](#34-data-provenance)
- [4 LLM for Data Management](#4-llm-for-data-management)
  - [4.1 LLM for Data Manipulation](#41-llm-for-data-manipulation)
  - [4.2 LLM for Data System Optimization](#42-llm-for-data-system-optimization)
- [5 LLM as Data Analyst](#5-llm-as-data-analyst)
   - [5.1 LLM for Structured Data Analysis](#51-llm-for-structured-data-analysis)
   - [5.2 LLM for Semi-Structured Data Analysis](#52-llm-for-semistructured-data-analysis)
   - [5.3 LLM for unstructured Data Analysis](#53-llm-for-unstructured-data-analysis)
   - [5.4 LLM for Heterogeneous Data Analysis](#54-llm-for-heterogeneous-data-analysis)

  <!-- - [4.2 LLM for Data Analysis](#42-llm-for-data-analysis) -->

## Datasets

1. **CommonCrawl**: A massive web crawl dataset covering diverse languages and domains; widely used for LLM pretraining. [[Source](https://commoncrawl.org/latest-crawl)]

1. **The Stack**: A large-scale dataset of permissively licensed source code in multiple programming languages; used for code LLMs. [[HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2)]

1. **RedPajama**: A replication of LLaMA’s training data recipe with open datasets; spans web, books, arXiv, and more. [[Github](https://github.com/togethercomputer/RedPajama-Data)]

1. **SlimPajama-627B-DC**: A deduplicated and filtered subset of RedPajama (627B tokens); optimized for clean and efficient training. [[HuggingFace](https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC)]

1. **Alpaca-CoT**: Instruction-following dataset enhanced with Chain-of-Thought (CoT) reasoning prompts; used for dialogue fine-tuning. [[Github](https://github.com/PhoebusSi/Alpaca-CoT?tab=readme-ov-file)]

1. **LLaVA-Pretrain**: A multimodal dataset with image-text pairs for training visual language models like LLaVA. [[HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)]

1. **Wikipedia**: Structured and encyclopedic content; a foundational source for general-purpose language models. [[HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia)]

1. **C4**: A cleaned version of CommonCrawl data, widely used in models like T5 for high-quality web text. [[HuggingFace](https://huggingface.co/datasets/allenai/c4)]

1. **BookCorpus**: Contains free fiction books; often used to teach models long-form language understanding. [[HuggingFace](https://huggingface.co/datasets/bookcorpus/bookcorpus)]

1. **Arxiv**: Scientific paper corpus from arXiv, covering physics, math, CS, and more; useful for academic language modeling. [[HuggingFace](https://huggingface.co/datasets/arxiv-community/arxiv_dataset)]

1. **PubMed**: Biomedical literature dataset from the PubMed database; key resource for medical domain models. [[Source](https://pubmed.ncbi.nlm.nih.gov/download/)]

1. **StackExchange**: Community Q&A data covering domains like programming, math, philosophy, etc.; useful for QA and dialogue tasks. [[Source](https://archive.org/details/stackexchange)]

1. **OpenWebText2**: A high-quality open-source web text dataset based on URLs commonly cited on Reddit; GPT-style training corpus. [[Source](https://openwebtext2.readthedocs.io/en/latest/)]

1. **OpenWebMath**: A dataset of math questions and answers; designed to improve mathematical reasoning in LLMs. [[HuggingFace](https://huggingface.co/datasets/open-web-math/open-web-math)]

1. **Falcon-RefinedWeb**: Filtered web data used in training Falcon models; emphasizes data quality through rigorous preprocessing. [[HuggingFace](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)]

1. **CCI 3.0**: A large-scale multi-domain Chinese web corpus, suitable for training high-quality Chinese LLMs. [[HuggingFace](https://huggingface.co/datasets/BAAI/CCI3-Data)]

1. **OmniCorpus**: A unified multimodal dataset (text, image, audio) designed for general-purpose AI training. [[Github](https://github.com/OpenGVLab/OmniCorpus?tab=readme-ov-file)]

1. **WanJuan3.0**: A diverse and large-scale Chinese dataset including news, fiction, QA, and more; released by OpenDataLab. [[Source](https://opendatalab.org.cn/OpenDataLab/WanJuan3)]

   

## 0 Data Characteristics across LLM Stages

[**⬆️top**](#table-of-contents)

### Data for Pretraining

1. **OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents**  
   Hugo Laurençon, Lucile Saulnier, Léo Tronchon, et al. *NeurIPS 2023*. [[Paper](https://neurips.cc/virtual/2023/poster/73589 )] 
2. **Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books**  
   Yukun Zhu, Ryan Kiros, Richard Zemel, et al. *ICCV 2015*. [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)] 

### Data for Continual Pre-training

1. **MedicalGPT: Training Medical GPT Model**   
   Ming Xu. [[Github](https://github.com/shibing624/MedicalGPT)]
2. **BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark**   
   Dakuan Lu, Hengkui Wu, Jiaqing Liang, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2302.09432 )] 

### Data for Supervised Fine-Tuning (SFT)

#### General Instruction Following

1. **Free dolly: Introducing the world’s first truly open instruction-tuned llm**    
Mike Conover, Matt Hayes, Ankit Mathur, et al. *2023*. [[Source](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)]

#### Specific Domain Usage

1. **MedicalGPT: Training Medical GPT Model** [[Github](https://github.com/shibing624/MedicalGPT)]
2. **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**  
   Shengbin Yue, Wei Chen, Siyuan Wang, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2309.11325)]

### Data for Reinforcement Learning (RL)

#### RLHF

1. **MedicalGPT: Training Medical GPT Model** [[Github](https://github.com/shibing624/MedicalGPT)]
2. **UltraFeedback: Boosting Language Models with Scaled AI Feedback**  
   Ganqu Cui, Lifan Yuan, Ning Ding, et al. *ICML 2024*. [[Paper](https://arxiv.org/abs/2310.01377)]

#### RoRL

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   DeepSeek-AI. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2501.12948)]
2. **Kimi k1.5: Scaling Reinforcement Learning with LLMs**  
   Kimi Team. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2501.12599)]

### Data for Retrieval-Augmented Generation (RAG)

1. **DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue**  
   Feiyuan Zhang, Dezhi Zhu, James Ming, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.13847)]
2. **Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation**  
   Junde Wu, Jiayuan Zhu, Yunli Qi, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2408.04187)]
3. **ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization**  
   Yunxiao Shi, Xing Zi, Zijing Shi, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.06683)]
4. **PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents**  
   Saber Zerhoudi, Michael Granitzer. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.09394)]
5. **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services** [[Paper](https://arxiv.org/abs/2309.11325)]

### Data for LLM Evaluation

1. **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**  
   Xiang Yue, Yuansheng Ni, Kai Zhang, et al. *CVPR 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)]
2. **LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models**  
   Haitao Li, You Chen, Qingyao Ai, et al. *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20288)]
3. **What disease does this patient have? a large-scale open domain question answering dataset from medical exams**   
   Di Jin, Eileen Pan, Nassim Oufattole, et al. *AAAI 2021*. [[Paper](https://arxiv.org/abs/2009.13081)]
4. **Evaluating Large Language Models Trained on Code**  
   Mark Chen, Jerry Tworek, Heewoo Jun, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2107.03374)]

### Data for LLM Agents

1. **STeCa: Step-level Trajectory Calibration for LLM Agent Learning**  
   Hanlin Wang, Jian Wang, Chak Tou Leong, Wenjie Li. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.14276)]
2. **Large Language Model-Based Agents for Software Engineering: A Survey**  
   Junwei Liu, Kaixin Wang, Yixuan Chen, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.02977)]
3. **Advancing LLM Reasoning Generalists with Preference Trees**  
   Lifan Yuan, Ganqu Cui, Hanbin Wang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2404.02078)]
4. **Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents**  
   Zhengliang Shi, Shen Gao, Lingyong Yan, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2405.16533)]
5. **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations**  
   Ning Ding, Yulin Chen, Bokai Xu, et al. *EMNLP 2023*. [[Paper](https://aclanthology.org/2023.emnlp-main.183/)]

## 1 Data Processing for LLM

[⬆️top](#table-of-contents)

### 1.1 Data Acquisition

#### Data Sources

##### Public Data

1. **Project Gutenberg**: A large collection of free eBooks from the public domain; supports training language models on long-form literary text. [[Source](https://www.gutenberg.org/)]
2. **Open Library**: A global catalog of books with metadata and some open-access content; useful for multilingual and knowledge-enhanced language modeling. [[Source](https://openlibrary.org/)]
3. **GitHub**: The world’s largest open-source code hosting platform; supports training models for code generation and understanding. [[Source](https://github.com/)]
4. **GitLab**: A DevOps platform for hosting both private and open-source projects; provides high-quality programming and documentation data. [[Source]( https://gitlab.com/)]
5. **Bitbucket**: A source code hosting platform by Atlassian; suitable for mining enterprise-level software development data. [[Source](https://bitbucket.org/product/)] 
6. **CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages**  
   Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, et al. *LREC-COLING 2024.* [[Paper](https://aclanthology.org/2024.lrec-main.377.pdf)]
7. **The Stack: 3 TB of permissively licensed source code**  
   Denis Kocetkov, Raymond Li, Loubna Ben Allal, et al. *arXiv 2022*. [[Paper](https://arxiv.org/abs/2211.15533)]
8. **mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer**  
   Linting Xue, Noah Constant, Adam Roberts, et al. *NAACL 2021.* [[Paper](https://aclanthology.org/2021.naacl-main.41.pdf)]
9. **Exploring the limits of transfer learning with a unified text-to-text transformer**  
     Colin Raffel, Noam Shazeer, Adam Roberts, et al. *JMLR 2020.* [[Paper](https://arxiv.org/abs/1910.10683)]
10. **CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**  
    Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, et al. *arXiv 2019*. [[Paper](https://arxiv.org/abs/1909.09436)]
11. **Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books** [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)] 



#### Data Acquisition Methods

##### Website Crawling

1. **Beautiful Soup**: A Python-based library for parsing HTML and XML documents; supports extracting structured information from static web pages. [[Source](https://beautiful-soup-4.readthedocs.io/en/latest/)]
2. **Selenium**: A browser automation tool that enables interaction with dynamic web pages; suitable for scraping JavaScript-heavy content. [[Github]( https://github.com/seleniumhq/selenium)]
3. **Playwright**: A browser automation framework developed by Microsoft; supports multi-browser environments and is ideal for high-quality, concurrent web scraping tasks. [[Source](https://playwright.dev/)]
4. **Puppeteer**: A Node.js library that provides a high-level API to control headless Chrome or Chromium; useful for scraping complex pages, taking screenshots, or generating PDFs. [[Source](https://pptr.dev/)]
5. **An Empirical Comparison of Web Content Extraction Algorithms**  
   Janek Bevendorff, Sanket Gupta, Johannes Kiesel, Benno Stein. *SIGIR 2023*. [[Paper](https://dl.acm.org/doi/10.1145/3539618.3591920)]
6. **Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction**  
   Adrien Barbaresi. *ACL 2021 Demo*. [[Paper](https://aclanthology.org/2021.acl-demo.15/)]
7. **Fact or Fiction: Content Classification for Digital Libraries**  
   Aidan Finn, N. Kushmerick, Barry Smyth. *DELOS Workshops / Conferences 2001.* [[Paper](https://www.semanticscholar.org/paper/Fact-or-Fiction%3A-Content-Classification-for-Digital-Finn-Kushmerick/73ccd5c477b37a082f66557a1793852d405e4b6d)]

##### Layout Analysis

1. **PaddleOCR**: An open-source Optical Character Recognition (OCR) toolkit based on the PaddlePaddle deep learning framework; supports multilingual text detection and recognition, ideal for extracting text from images and document layout analysis. [[Github](https://github.com/paddlepaddle/paddleocr)]
2. **YOLOv10: Real-Time End-to-End Object Detection**  
   Ao Wang, Hui Chen, Lihao Liu, et al. *NeurIPS 2024.* [[Paper](https://arxiv.org/abs/2405.14458)]
3. **UMIE: Unified Multimodal Information Extraction with Instruction Tuning**  
   Lin Sun, Kai Zhang, Qingyuan Li, Renze Lou. *AAAI 2024.* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29873)]
4. **ChatEL: Entity linking with chatbots**  
   Yifan Ding, Qingkai Zeng, Tim Weninger. *LREC | COLING 2024*. [[Paper](https://aclanthology.org/2024.lrec-main.275/)]
5. **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**  
   Haoran Wei, Lingyu Kong, Jinyue Chen, et al. *ECCV 2024.* [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73235-5_23)]
6. **General OCR Theory: Towards OCR - 2.0 via a Unified End - to - end Model**  
   Haoran Wei, Chenglong Liu, Jinyue Chen, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2409.01704v1)]
7. **Focus Anywhere for Fine-grained Multi-page Document Understanding**  
   Chenglong Liu, Haoran Wei, Jinyue Chen, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14295)]
8. **MinerU: An Open-Source Solution for Precise Document Content Extraction**  
   Bin Wang, Chao Xu, Xiaomeng Zhao, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2409.18839)]
9. **WebIE: Faithful and Robust Information Extraction on the Web**  
   Chenxi Whitehouse, Clara Vania, Alham Fikri Aji, et al. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-long.428/)]
10. **ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking**  
    Tom Ayoola, Shubhi Tyagi, Joseph Fisher, et al. *NAACL 2022 Industry Track.* [[Paper](https://aclanthology.org/2022.naacl-industry.24.pdf)]
11. **Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction**  
    Keshav Kolluru, Muqeeth Mohammed, Shubham Mittal, et al. *ACL 2022.* [[Paper](https://aclanthology.org/2022.acl-long.179/)]
12. **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**  
    Yupan Huang, Tengchao Lv, Lei Cui, et al. *ACM Multimedia 2022.* [[Paper](https://arxiv.org/abs/2204.08387)]
13. **Learning Transferable Visual Models From Natural Language Supervision**  
    Alec Radford, Jong Wook Kim, Chris Hallacy, et al. *ICML 2021.* [[Paper](https://proceedings.mlr.press/v139/radford21a)]
14. **Tesseract: an open-source optical character recognition engine**  
    Anthony Kay. *Linux Journal, Volume 2007*. [[Paper](https://dl.acm.org/doi/10.5555/1288165.1288167)]



### 1.2 Data Deduplication

[⬆️top](#table-of-contents)

1. **Analysis of the Reasoning with Redundant Information Provided Ability of Large Language Models**  
   Wenbei Xie. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2310.04039v1)]
2. **Scaling Laws and Interpretability of Learning from Repeated Data**  
   Danny Hernandez, Tom Brown, Tom Conerly, et al. *arXiv 2022.* [[Paper](https://arxiv.org/abs/2205.10487)]

#### Exact Substring Matching

1. **BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and     Deduplication by Introducing a Competitive Large Language Model Baseline**    
   Guosheng Dong, Da Pan, Yiding Sun, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2408.15079)]
2. **Deduplicating Training Data Makes Language Models Better**    
   Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al. *ACL 2022.* [[Paper](https://arxiv.org/abs/2107.06499)]
3. **Suffix arrays: a new method for on-line string searches**  
   Udi Manber, Gene Myers. *SIAM Journal on Computing 1993.* [[Paper](https://doi.org/10.1137/0222058)]

#### Approximate Hashing-based Deduplication

1. **BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and     Deduplication by Introducing a Competitive Large Language Model Baseline** [[Paper](https://arxiv.org/abs/2408.15079)]
2. **LSHBloom: Memory-efficient, Extreme-scale Document Deduplication**  
   Arham Khan, Robert Underwood, Carlo Siebenschuh, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2411.04257)]
3. **SimiSketch: Efficiently Estimating Similarity of streaming Multisets**   
   Fenghao Dong, Yang He, Yutong Liang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2405.19711)] 
4. **DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication**  
   Igor Nunes, Mike Heddes, Pere Vergés, et al. *KDD 2023.* [[Paper](https://doi.org/10.1145/3580305.3599314)]
5. **Formalizing BPE Tokenization**  
   Martin Berglund (Umeå University), Brink van der Merwe (Stellenbosch University). *NCMA 2023*. [[Paper](https://arxiv.org/abs/2309.08715)]
6. **SlimPajama-DC: Understanding Data Combinations for LLM Training**  
   Zhiqiang Shen, Tianhua Tao, Liqun Ma, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.10818)]
7. **Deduplicating Training Data Makes Language Models Better** [[Paper](https://arxiv.org/abs/2107.06499)]
8. **Noise-Robust De-Duplication at Scale**  
   Emily Silcock, Luca D'Amico-Wong, Jinglin Yang, Melissa Dell. *arXiv 2022.* [[Paper](https://arxiv.org/abs/2210.04261)]
9. **In Defense of Minhash over Simhash**  
   Anshumali Shrivastava, Ping Li. *AISTATS 2014.* [[Paper](https://proceedings.mlr.press/v33/shrivastava14.html)]
10. **Similarity estimation techniques from rounding algorithms**  
    Moses S. Charikar. *STOC 2002.* [[Paper](https://doi.org/10.1145/509907.509965)]
11. **On the Resemblance and Containment of Documents**  
    A. Broder. *Compression and Complexity of SEQUENCES 1997.* [[Paper](https://doi.org/10.1109/SEQUEN.1997.666900)]

#### Approximate Frequency-based Down-Weighting

1. **SoftDedup: an Efficient Data Reweighting Method for Speeding Up Language Model Pre-training**  
   Nan He, Weichen Xiong, Hanwen Liu, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.220/)]

#### Embedding-Based Clustering

1. **FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication**  
   Eric Slyman, Stefan Lee, Scott Cohen, Kushal Kafle. *CVPR 2024.* [[Paper](https://arxiv.org/abs/2404.16123)]
2. **Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters**  
   Amro Abbas, Evgenia Rusak, Kushal Tirumala, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.04578)]
3. **D4: Improving LLM Pretraining via Document De-Duplication and Diversification**  
   Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos. *NeurIPS 2023.* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)]
4. **SemDeDup: Data-efficient learning at web-scale through semantic deduplication**  
   Amro Abbas, Kushal Tirumala, Dániel Simig, et al. *ICLR 2023.* [[Paper](https://iclr.cc/virtual/2023/13610)]
5. **OPT: Open Pre-trained Transformer Language Models**  
   Susan Zhang, Stephen Roller, Naman Goyal, et al. *arXiv 2022.* [[Paper](https://arxiv.org/abs/2205.01068v4)]
6. **Learning Transferable Visual Models From Natural Language Supervision** [[Paper](https://proceedings.mlr.press/v139/radford21a)]
7. **OpenCLIP**     
   Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, et al. *2021*. [[Paper](https://doi.org/10.5281/zenodo.5143773)]
8. **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs**  
   Christoph Schuhmann, Richard Vencu, Romain Beaumont, et al. *NeurIPS 2021.* [[Paper](https://doi.org/10.48550/arXiv.2111.02114)]

#### Non-Text Data Deduplication

1. **DataComp: In search of the next generation of multimodal datasets**  
   Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2304.14108)]
2. **SemDeDup: Data-efficient learning at web-scale through semantic deduplication** [[Paper](https://iclr.cc/virtual/2023/13610)]
3. **Learning Transferable Visual Models From Natural Language Supervision** [[Paper](https://proceedings.mlr.press/v139/radford21a)]
4. **Contrastive Learning with Large Memory Bank and Negative Embedding Subtraction for Accurate Copy Detection**  
   Shuhei Yokoo. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2112.04323)]



### 1.3 Data Filtering

[⬆️top](#table-of-contents)

#### Sample-level Filtering

##### (1) Statistical Evaluation

1. **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models**  
   Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, et al. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/31214)]
2. **Data-efficient Fine-tuning for LLM-based Recommendation**  
   Xinyu Lin, Wenjie Wang, Yongqi Li, et al. *SIGIR 2024.* [[Paper](https://arxiv.org/abs/2401.17197)]
3. **SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning**  
   Yexiao He, Ziyao Wang, Zheyu Shen, et al. *NeurIPS 2024.* [[Paper](https://arxiv.org/abs/2405.00705)]
4. **SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models**  
   Yu Yang, Siddhartha Mishra, Jeffrey Chiang, et al. *NeurIPS 2024.* [[Paper](https://neurips.cc/virtual/2024/poster/95679)]
5. **Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters** [[Paper](https://doi.org/10.48550/arXiv.2401.04578)]
6. **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions**  
   Can Xu, Qingfeng Sun, Kai Zheng, et al. *ICLR 2024.* [[Paper](https://iclr.cc/virtual/2024/poster/19164)]
7. **Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning**  
   Ming Li, Yong Zhang, Shwai He, et al. *ACL 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.00530)]
8. **Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models**  
   Dheeraj Mekala, Alex Nguyen, Jingbo Shang. *ACL 2024*. [[Paper](https://aclanthology.org/2024.findings-acl.623/)]
9. **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**  
   Luca Soldaini, Rodney Kinney, Akshita Bhagia, et al. *ACL 2024*. [[Paper](https://arxiv.org/abs/2402.00159)]
10. **From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning**  
    Ming Li, Yong Zhang, Zhitao Li, et al. *NAACL 2024*. [[Paper](https://arxiv.org/abs/2308.12032)]
11. **Improving Pretraining Data Using Perplexity Correlations**  
    Tristan Thrush, Christopher Potts, Tatsunori Hashimoto. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.05816)]
12. **Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs**  
    The Mosaic Research Team. *2023*. [[Paper](https://www.databricks.com/blog/mpt-7b)]
13. **Instruction Tuning with GPT-4**  
    Baolin Peng, Chunyuan Li, Pengcheng He, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2304.03277)]
14. **DINOv2: Learning Robust Visual Features without Supervision**  
    Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2304.07193)]
15. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**  
    Leo Gao, Stella Biderman, Sid Black, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2101.00027)]
16. **Language Models are Unsupervised Multitask Learners**  
    Alec Radford, Jeffrey Wu, Rewon Child, et al. *OpenAI blog 2019*. [[Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)]
17. **Bag of Tricks for Efficient Text Classification**  
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *EACL 2017.* [[Paper](https://aclanthology.org/E17-2068.pdf)]
18. **The Shapley Value: Essays in Honor of Lloyd S. Shapley**  
    A. E. Roth, Ed. *Cambridge: Cambridge University Press, 1988*. [[Source](https://www.cambridge.org/core/books/shapley-value/D3829B63B5C3108EFB62C4009E2B966E)]

##### (2) Model Scoring

1. **SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection**  
   Han Shen, Pin-Yu Chen, Payel Das, Tianyi Chen. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/29422)]
2. **SCAR: Data Selection via Style-Consistency-Aware Response Ranking for Efficient Instruction Tuning of Large Language Models**  
   Zhuang Li, Yuncheng Hua, Thuy-Trang Vu, et al. *ACL 2025.* [[Paper](https://arxiv.org/abs/2406.10882)] [[Github](https://github.com/zhuang-li/SCAR)]
3. **QuRating: Selecting High-Quality Data for Training Language Models**  
   Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen. *ICML 2024.* [[Paper](https://arxiv.org/abs/2402.09739)]
4. **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**  
   Wei Liu, Weihao Zeng, Keqing He, et al. *ICLR 2024.* [[Paper](https://arxiv.org/abs/2312.15685)]
5. **LAB: Large-Scale Alignment for ChatBots**  
   Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.01081)]
6. **Biases in Large Language Models: Origins, Inventory, and Discussion**  
   Roberto Navigli, Simone Conia, Björn Ross. *ACM JDIQ, 2023.* [[Paper](https://doi.org/10.1145/3597307)]

##### (3) Hybrid Methods

1. **Emergent and predictable memorization in large language models**  
   Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, et al. *NeurIPS 2023*. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667341?__cf_chl_tk=sWnInkGSOKRsrS.z3RwRKDT836eoSy1i.k5oxZcfDzA-1748509375-1.0.1.1-lmH0EWkZpuiyEr5uZPEd_C92GFkM6u6BY416q24qBww)]
2. **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**  
   Max Marion, Ahmet Üstün, Luiza Pozzobon, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.04564)]
3. **Instruction Mining: Instruction Data Selection for Tuning Large Language Models**  
   Yihan Cao, Yanbin Kang, Chi Wang, Lichao Sun. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2307.06290)]
4. **Llama 2: Open Foundation and Fine-Tuned Chat Models**  
   Hugo Touvron, Louis Martin, Kevin Stone, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2307.09288)]
5. **MoDS: Model-oriented Data Selection for Instruction Tuning**  
   Qianlong Du, Chengqing Zong, Jiajun Zhang. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2311.15653)]
6. **Economic Hyperparameter Optimization With Blended Search Strategy**  
   Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. *ICLR 2021.* [[Paper](https://iclr.cc/virtual/2021/poster/3052)]
7. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   Jacob Devlin, Ming-Wei Chang, Kenton Lee, et al. *NAACL 2019.* [[Paper](https://aclanthology.org/N19-1423.pdf)]
8. **Active Learning for Convolutional Neural Networks: A Core-Set Approach**  
   Ozan Sener, Silvio Savarese. *ICLR 2018.* [[Paper](https://doi.org/10.48550/arXiv.1708.00489)]

#### Content-level Filtering

1. **spaCy**: An industrial-strength Natural Language Processing (NLP) library that supports tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and more; well-suited for fast and accurate text processing and information extraction. [[Source](https://spacy.io/)]
2. **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**  
   Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. *ICLR 2025*. [[Paper](https://arxiv.org/abs/2408.06072)]
3. **HunyuanVideo: A Systematic Framework For Large Video Generative Models**  
   Weijie Kong, Qi Tian, Zijian Zhang, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2412.03603v6)]
4. **Wan: Open and Advanced Large-Scale Video Generative Models**  
   Team Wan et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.20314)]
5. **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**   
   Hang Zhang, Xin Li, Lidong Bing. *EMNLP 2023 (System Demonstrations)*. [[Paper](https://arxiv.org/abs/2306.02858)]
6. **Analyzing Leakage of Personally Identifiable Information in Language Models**  
   Nils Lukas, Ahmed Salem, Robert Sim, et al. *IEEE S&P 2023.* [[Paper](https://arxiv.org/abs/2302.00539)]
7. **DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4**  
   Zhengliang Liu, Yue Huang, Xiaowei Yu, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2303.11032)]
8. **Baichuan 2: Open Large-scale Language Models**  
   Aiyuan Yang, Bin Xiao, Bingning Wang, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.10305)]
9. **Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives**  
   Haoning Wu, Erli Zhang, Liang Liao, et al. *arXiv 2022*. [[Paper](https://arxiv.org/abs/2211.04894)]
10. **YOLOX: Exceeding YOLO Series in 2021**  
      Zheng Ge, Songtao Liu, Feng Wang, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2107.08430)]
11. **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs** [[Paper](https://doi.org/10.48550/arXiv.2111.02114)]
12. **FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP**  
    Alan Akbik, Tanja Bergmann, Duncan Blythe, et al. *NAACL 2019 Demos.* [[Paper](https://aclanthology.org/N19-4010/)]

### 1.4 Data Selection

[⬆️top](#table-of-contents)

1. **A Survey on Data Selection for Language Models**  
   Alon Albalak, Yanai Elazar, Sang Michael Xie, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2402.16827)]

2. **A Survey on Data Selection for LLM Instruction Tuning**  
   Jiahao Wang, Bolin Zhang, Qianlong Du, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2402.05123)]

#### Similarity-based Data Selection

1. **spaCy**:  [[Source](https://spacy.io/)]
2. **Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis**  
   Ruiyang Qin, Jun Xia, Zhenge Jia, et al. *DAC 2024.* [[Paper](https://doi.org/10.1145/3649329.3655665)]
3. **CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training**  
   David Brandfonbrener, Hanlin Zhang, Andreas Kirsch, et al. *NeurIPS 2024.* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0f25f0a63cc544d506e4c1374a3c807-Abstract-Conference.html)]
4. **Efficient Continual Pre-training for Building Domain Specific Large Language Models**  
   Yong Xie, Karan Aggarwal, Aitzaz Ahmad. *Findings of ACL 2024*. [[Paper](https://aclanthology.org/2024.findings-acl.606/)]
5. **Data Selection for Language Models via Importance Resampling**  
   Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang. *NeurIPS 2023.* [[Paper](https://doi.org/10.48550/arXiv.2302.03169)]

#### Optimization-based Data Selection

1. **DSDM: model-aware dataset selection with datamodels**  
   Logan Engstrom, Axel Feldmann, Aleksander Mądry. *ICML 2024.* [[Paper](https://dl.acm.org/doi/10.5555/3692070.3692568)]
2. **LESS: Selecting Influential Data for Targeted Instruction Tuning**  
   Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, et al. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.04333)]
3. **TSDS: Data Selection for Task-Specific Model Finetuning**  
   Zifan Liu, Amin Karbasi, Theodoros Rekatsinas. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.11303)]
4. **Datamodels: Understanding Predictions with Data and Data with Predictions**  
   Andrew Ilyas, Sung Min Park, Logan Engstrom, et al. *ICML 2022.* [[Paper](https://proceedings.mlr.press/v162/ilyas22a.html)]

#### Model-based Data Selection

1. **Autonomous Data Selection with Language Models for Mathematical Texts**  
   Yifan Zhang, Yifan Luo, Yang Yuan, et al. *ICLR 2024.* [[Paper](https://iclr.cc/virtual/2024/22423)]



### 1.5 Data Mixing

[⬆️top](#table-of-contents)

1. **Mixtera: A Data Plane for Foundation Model Training** Maximilian Böther, Xiaozhe Yao, Tolga Kerimoglu, Dan Graur, Viktor Gsteiger, Ana Klimovic. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2502.19790)]
2. **Scalable Data Ablation Approximations for Language Models through Modular Training and Merging**  
   Clara Na, Ian Magnusson, Ananya Harsh Jha, et al. *EMNLP 2024.* [[Paper](https://arxiv.org/abs/2410.15661v1)]
3. **Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models**  
   Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, et al. *COLING 2024.* [[Paper](https://arxiv.org/abs/2403.03432v1)]

#### Heuristic Optimization

1. **BiMix: Bivariate Data Mixing Law for Language Model Pretraining**  
   Ce Ge, Zhijian Ma, Daoyuan Chen, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14908)]
2. **Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining**  
   Steven Feng, Shrimai Prabhumoye, Kezhi Kong, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.15285)]
3. **SlimPajama-DC: Understanding Data Combinations for LLM Training** [[Paper](https://arxiv.org/abs/2309.10818)]
4. **Evaluating Large Language Models Trained on Code** [[Paper](https://arxiv.org/abs/2107.03374)]
5. **Exploring the limits of transfer learning with a unified text-to-text transformer** [[Paper](https://arxiv.org/abs/1910.10683v4)]
6. **CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge**  
   Alon Talmor, Jonathan Herzig, Nicholas Lourie, et al. *NAACL 2019*. [[Paper](https://arxiv.org/abs/1811.00937)]
7. **A mathematical theory of communication**  
   C. E. Shannon. *The Bell system technical journal 1948*. [[Paper](https://ieeexplore.ieee.org/document/6773024)]

#### Bilevel Optimization

1. **ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting**  
   Rui Pan, Jipeng Zhang, Xingyuan Pan, et al. *ACL 2025.* [[Paper](https://arxiv.org/abs/2406.19976)]
2. **DoGE: Domain Reweighting with Generalization Estimation**  
   Simin Fan, Matteo Pagliardini, Martin Jaggi. *ICML 2024.* [[Paper](https://icml.cc/virtual/2024/poster/34869)]
3. **An overview of bilevel optimization**  
   Benoît Colson, Patrice Marcotte, Gilles Savard. *AOR 2007.* [[Paper](https://link.springer.com/article/10.1007/s10479-007-0176-2)]

#### Distributionally Robust Optimization

1. **Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval**  
   Guangyuan Ma, Yongliang Ma, Xing Wu, et al. *AAAI 2025.* [[Paper](https://arxiv.org/abs/2408.10613)]
2. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**  
   Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. *NeurIPS 2023.* [[Paper](https://arxiv.org/abs/2305.10429)]
3. **Qwen Technical Report**  
   Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.16609v1)]

#### Model-Based Optimization

1. **RegMix: Data Mixture as Regression for Language Model Pre-training**  
   Qian Liu, Xiaosen Zheng, Niklas Muennighoff, et al. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/30960)]
2. **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**  
   Jiasheng Ye, Peiju Liu, Tianxiang Sun, et al. *ICLR 2025.* [[Paper](https://arxiv.org/abs/2403.16952)]
3. **CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models**  
   Jiawei Gu, Zacc Yang, Chuanghao Ding, et al. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.903)]
4. **TinyLlama: An Open-Source Small Language Model**  
   Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2401.02385)]
5. **BiMix: Bivariate Data Mixing Law for Language Model Pretraining** [[Paper](https://arxiv.org/abs/2405.14908)]
6. **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models**  
   Haoran Que, Jiaheng Liu, Ge Zhang, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.01375)]
7. **Data Proportion Detection for Optimized Data Management for Large Language Models**  
   Hao Liang, Keshi Zhao, Yajie Yang, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2409.17527)]
8. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining** [[Paper](https://arxiv.org/abs/2305.10429)]
9. **Training compute-optimal large language models**  
   Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *NeurIPS 2022.* [[Paper](https://dl.acm.org/doi/10.5555/3600270.3602446)]
10. **LightGBM: a highly efficient gradient boosting decision tree**  
    Guolin Ke, Qi Meng, Thomas Finley, et al. *NeurIPS 2017.* [[Paper](https://dl.acm.org/doi/10.5555/3294996.3295074)]



### 1.6 Data Distillation and Synthesis

[⬆️top](#table-of-contents)

1. **How to Synthesize Text Data without Model Collapse?**  
   Xuekai Zhu, Daixuan Cheng, Hengli Li, et al. *ICML 2025*. [[Paper](https://arxiv.org/abs/2412.14689)]
2. **Differentially Private Synthetic Data via Foundation Model APIs 2: Text**  
   Chulin Xie, Zinan Lin, Arturs Backurs, et al. *ICML 2024.* [[Paper](https://arxiv.org/abs/2403.01749v2)]
3. **LLM See, LLM Do: Leveraging Active Inheritance to Target Non-Differentiable Objectives**  
   Luísa Shimabucoro, Sebastian Ruder, Julia Kreutzer, et al. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.521)]
4. **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions** [[Paper](https://iclr.cc/virtual/2024/poster/19164)]
5. **Augmenting Math Word Problems via Iterative Question Composing**  
   Haoxiong Liu, Yifan Zhang, Yifan Luo, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.09003)]

#### Knowledge Distillation

1. **Multistage Collaborative Knowledge Distillation from a Large Language Model for Semi-Supervised Sequence Generation**   
   Jiachen Zhao, Wenlong Zhao, Andrew Drozdov, et al. *ACL 2024*. [[Paper](https://arxiv.org/abs/2311.08640)]
2. **PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning**  
   Xuekai Zhu, Biqing Qi, Kaiyan Zhang, et al. *NAACL 2024*. [[Paper](https://arxiv.org/abs/2305.13888)]
3. **Knowledge Distillation Using Frontier Open-source LLMs: Generalizability and the Role of Synthetic Data**   
   Anup Shirgaonkar, Nikhil Pandey, Nazmiye Ceren Abay, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.18588)]
4. **Training Verifiers to Solve Math Word Problems**  
   Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2110.14168)]
5. **Dialogue chain-of-thought distillation for commonsense-aware conversational agents**  
   Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2310.09343)]
6. **MCC-KD: Multi-CoT consistent knowledge distillation**  
   Hongzhan Chen, Siyue Wu, Xiaojun Quan, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2310.14747)]
7. **Large language models are reasoning teachers**  
   Namgyu Ho, Laura Schmid, Se-Young Yun. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2212.10071)]
8. **Leveraging training data in few-shot prompting for numerical reasoning**  
   Zhanming Jie, Wei Lu. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2305.18170)]
9. **Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks**  
   Minki Kang, Seanie Lee, Jinheon Baek, et al. *NeurIPS 2023*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html)]
10. **Symbolic chain-of-thought distillation: Small models can also "think" step-by-step**  
    Liunian Harold Li, Jack Hessel, Youngjae Yu, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2306.14050)]
11. **Explanations from large language models make small reasoners better**  
    Shiyang Li, Jianshu Chen, Yelong Shen, et al. *arXiv 2022*. [[Paper](https://arxiv.org/abs/2210.06726)]
12. **Distilling reasoning capabilities into smaller language models**  
    Kumar Shridhar, Alessandro Stolfo, Mrinmaya Sachan. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2212.00193)]
13. **SCOTT: Self-consistent chain-of-thought distillation**  
    Peifeng Wang, Zhengyang Wang, Zheng Li, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2305.01879)]
14. **Democratizing reasoning ability: Tailored learning from large language model**  
    Zhaoyang Wang, Shaohan Huang, Yuxuan Liu, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2310.13332)]

#### Pre-training Data Augmentation

1. **BERT-Tiny-Chinese**: A lightweight Chinese BERT pre-trained model released by CKIP Lab, with a small number of parameters; suitable for use as an encoder in pre-training data augmentation tasks to enhance efficiency for compact models. [[Source](https://huggingface.co/ckiplab/bert-tiny-chinese)]
2. **Case2Code: Scalable Synthetic Data for Code Generation**   
   Yunfan Shao, Linyang Li, Yichuan Ma, et al. *COLING 2025*. [[Paper](https://aclanthology.org/2025.coling-main.733/)]
3. **Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages**  
   Zui Chen, Tianqiao Liu, Mi Tian, et al. *ICLR 2025*. [[Paper](https://arxiv.org/abs/2501.14002)]
4. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**  
   Kun Zhou, Beichen Zhang, Jiapeng Wang, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14365)]
5. **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**  
   Bin Xiao, Haiping Wu, Weijian Xu, et al. *CVPR 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)]
6. **DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models**  
   Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, et al. *CVPR 2024*. [[Paper](https://arxiv.org/abs/2405.14881)]
7. **Magicoder: Empowering Code Generation with OSS-Instruct**   
   Yuxiang Wei, Zhe Wang, Jiawei Liu, et al. *ICML 2024*. [[Paper](https://arxiv.org/abs/2312.02120)]
8. **Instruction Pre-Training: Language Models are Supervised Multitask Learners**  
   Daixuan Cheng, Yuxian Gu, Shaohan Huang, et al. *EMNLP 2024*. [[Paper](https://arxiv.org/abs/2406.14491)]
9. **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research** [[Paper](https://arxiv.org/abs/2402.00159)]
10. **Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling**   
    Pratyush Maini, Skyler Seto, Richard Bai, et al. *ACL 2024*. [[Paper](https://aclanthology.org/2024.acl-long.757/)]
11. **VeCLIP: Improving CLIP Training via Visual-Enriched Captions**  
    Zhengfeng Lai, Haotian Zhang, Bowen Zhang, et al. *ECCV 2024*. [[Paper](https://dl.acm.org/doi/10.1007/978-3-031-72946-1_7)]
12. **Diffusion Models and Representation Learning: A Survey**  
    Michael Fuest, Pingchuan Ma, Ming Gui, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.00783)]
13. **CtrlSynth: Controllable Image Text Synthesis for Data-Efficient Multimodal Learning**  
    Qingqing Cao, Mahyar Najibi, Sachin Mehta. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.11963)]
14. **Qwen2 Technical Report**  
    An Yang, Baosong Yang, Binyuan Hui, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.10671)]
15. **TinyLlama: An Open-Source Small Language Model** [[Paper](https://arxiv.org/abs/2401.02385)]
16. **On the Diversity of Synthetic Data and its Impact on Training Large Language Models**  
    Hao Chen, Abdul Waheed, Xiang Li, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.15226)]
17. **Towards Effective and Efficient Continual Pre-training of Large Language Models**  
    Jie Chen, Zhipeng Chen, Jiapeng Wang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.18743)]
18. **Improving CLIP Training with Language Rewrites**  
    Lijie Fan, Dilip Krishnan, Phillip Isola, et al. *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2305.20088)]
19. **Effective Data Augmentation With Diffusion Models**  
    Brandon Trabucco, Kyle Doherty, Max Gurinas, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2302.07944)]
20. **Mistral 7B**  
    Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2310.06825)]
21. **Llama 2: Open Foundation and Fine-Tuned Chat Models** [[Paper](https://arxiv.org/abs/2307.09288)]
22. **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**  
    Dustin Podell, Zion English, Kyle Lacey, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2307.01952)]
23. **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**  
    Jesse Dodge, Maarten Sap, Ana Marasović, et al. *EMNLP 2021*. [[Paper](https://arxiv.org/abs/2104.08758)]
24. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling** [[Paper](https://arxiv.org/abs/2101.00027)]
25. **First Steps of an Approach to the ARC Challenge based on Descriptive Grid Models and the Minimum Description Length Principle**  
    Sébastien Ferré (Univ Rennes, CNRS, IRISA). *arXiv 2021*. [[Paper](https://arxiv.org/abs/2112.00848)]
26. **TinyBERT: Distilling BERT for Natural Language Understanding**  
    Xiaoqi Jiao, Yichun Yin, Lifeng Shang, et al. *Findings of EMNLP 2020*. [[Paper](https://arxiv.org/abs/1909.10351)]
27. **HellaSwag: Can a Machine Really Finish Your Sentence?**  
    Rowan Zellers, Ari Holtzman, Yonatan Bisk, et al. *ACL 2019*. [[Paper](https://arxiv.org/abs/1905.07830)]

#### SFT Data Augmentation

1. **Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning**  
   Yiming Huang, Xiao Liu, Yeyun Gong, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.02333)]
2. **Augmenting Math Word Problems via Iterative Question Composing** [[Paper](https://doi.org/10.48550/arXiv.2401.09003)]
3. **AgentInstruct: Toward Generative Teaching with Agentic Flows**  
   Arindam Mitra, Luciano Del Corro, Guoqing Zheng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2407.03502)]
4. **Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models**  
   Haoran Li, Qingxiu Dong, Zhengyang Tang, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.13064)]
5. **Self-Instruct: Aligning Language Models with Self-Generated Instructions**  
   Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, et al. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-long.754)]

#### SFT Reasoning Data Augmentation

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** [[Paper](https://arxiv.org/abs/2501.12948)]
2. **LIMO: Less is More for Reasoning**  
   Yixin Ye, Zhen Huang, Yang Xiao, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.03387)]
3. **LLMs Can Easily Learn to Reason from Demonstrations: Structure, Not Content, Is What Matters!**  
   Dacheng Li, Shiyi Cao, Tyler Griggs, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.07374)]
4. **Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search**  
   Maohao Shen, Guangtao Zeng, Zhenting Qi, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.02508)]
5. **Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling**  
   Zhenyu Hou, Xin Lv, Rui Lu, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.11651)]
6. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data**  
   Yinya Huang, Xiaohan Lin, Zhengying Liu, et al. *ICLR 2024.* [[Paper](https://arxiv.org/abs/2402.08957v3)]
7. **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**  
   Peiyi Wang, Lei Li, Zhihong Shao, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.510)]
8. **NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions**   
   Jia Li, Edward Beeching, Lewis Tunstall, et al. *2024*. [[Paper](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)]
9. **QwQ: Reflect Deeply on the Boundaries of the Unknown**   
   Qwen Team. *2024*. [[Source](https://qwenlm.github.io/blog/qwq-32b-preview/)]
10. **Let's Verify Step by Step**  
    Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2305.20050)]

#### Reinforcement Learning

1. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
   Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. *NeurIPS 2023*. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3668142)]
2. **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**  
   Yuntao Bai, Andy Jones, Kamal Ndousse, et al. *arXiv 2022.* [[Paper](https://doi.org/10.48550/arXiv.2204.05862)]

#### Retrieval-Augmentation Generation

1. **Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data**  
   Shenglai Zeng, Jiankun Zhang, Pengfei He, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.14773)]



### 1.7 End-to-End Data Processing Pipelines

[⬆️top](#table-of-contents)

#### 1.7.1 Typical data processing frameworks

1. **Mixtera: A Data Plane for Foundation Model Training**     
   Maximilian Böther, Xiaozhe Yao, Tolga Kerimoglu, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2502.19790)]
2. **Data-Juicer: A One-Stop Data Processing System for Large Language Models**  
   Daoyuan Chen, Yilun Huang, Zhijian Ma, et al. *SIGMOD 2024.* [[Paper](https://doi.org/10.1145/3626246.3653385)]
3. **An Integrated Data Processing Framework for Pretraining Foundation Models**  
   Yiding Sun, Feng Wang, Yutao Zhu, et al. *SIGIR 2024.* [[Paper](https://doi.org/10.1145/3626772.3657671)]
4. **Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models**  
   Hyunbyung Park, Sukyung Lee, Gyoungjin Gim, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2403.19340v1)]

#### 1.7.2 Typical data pipelines

1. **Common Crawl**: A large-scale publicly accessible web crawl dataset that provides massive raw webpages and metadata. It serves as a crucial raw data source in typical pretraining data pipelines, where it undergoes multiple processing steps such as cleaning, deduplication, and formatting to produce high-quality corpora for downstream model training. [[Source](https://commoncrawl.org/)]
2. **The RefinedWeb dataset for falcon LLM: outperforming curated corpora with web data only**  
   Guilherme Penedo, Quentin Malartic, Daniel Hesslow, et al. *NeurIPS 2023.* [[Paper](https://dl.acm.org/doi/10.5555/3666122.3669586)]
3. **Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction** [[Paper](https://aclanthology.org/2021.acl-demo.15.pdf)]
4. **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**  
   Jack W. Rae, Sebastian Borgeaud, Trevor Cai, et al. *arXiv 2021.* [[Paper](https://arxiv.org/abs/2112.11446v2)]
5. **CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data**  
   Guillaume Wenzek, Marie - Anne Lachaux, Alexis Conneau, et al. *LREC 2020.* [[Paper](https://aclanthology.org/2020.lrec-1.494/)]
6. **Exploring the limits of transfer learning with a unified text-to-text transformer** [[Paper](https://arxiv.org/abs/1910.10683v4)]  
7. **Bag of Tricks for Efficient Text Classification** [[Paper](https://aclanthology.org/E17-2068.pdf)]

#### 1.7.3 Orchestration of data pipelines

1. **Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development** [[Paper](https://arxiv.org/abs/2407.11784v2)]



## 2 Data Storage for LLM

[⬆️top](#table-of-contents)

### 2.1 Data Formats

#### Training Data Format

1. **TFRecord**: A binary data storage format recommended by TensorFlow, suitable for efficient storage and reading of large-scale training data. [[Source](https://www.tensorflow.org/tutorials/load_data/tfrecord)]
2. **MindRecord**: An efficient data storage format used by MindSpore, supporting multi-platform data management. [[Source](https://www.mindspore.cn/)]
3. **tf.data.Dataset**: An abstract interface in TensorFlow representing collections of training data, enabling flexible data manipulation. [[Source](https://www.tensorflow.org/guide/data)]
4. **COCO JSON**: COCO JSON format uses structured JSON to store images and their corresponding labels, widely used in computer vision datasets. [[Source](https://cocodataset.org/)]

#### Model Data Format

1. **PyTorch-specific formats (.pt, .pth)**: PyTorch’s .pt and .pth formats are used to save model parameters and architecture, supporting model storage and loading. [[Source](https://pytorch.org/)]
2. **TensorFlow(SavedModel, .ckpt)**: TensorFlow’s SavedModel and checkpoint formats save complete model information, facilitating model reproduction and deployment. [[Source](https://www.tensorflow.org)]
3. **Hugging Face Transformers library**: Hugging Face offers a unified model format interface to facilitate saving and usage of various pretrained models. [[Source]( https://huggingface.co/)]
4. **Pickle (.pkl)**: Pickle format is used for serializing models and data, suitable for quick saving and loading. [[Source](https://docs.python.org/3/library/pickle.html)]
5. **ONNX**: An open cross-platform model format supporting model conversion and deployment across different frameworks. [[Source](https://onnx.ai)]
6. **An Empirical Study of Safetensors' Usage Trends and Developers' Perceptions**  
   Beatrice Casey, Kaia Damian, Andrew Cotaj, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.02170)]



### 2.2 Data Distribution

[⬆️top](#table-of-contents)

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** [[Paper](https://arxiv.org/abs/2501.12948)]
2. **CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl**  
   Ilya Ilyankou, Meihui Wang, Stefano Cavazzi, et al. *SIGSPATIAL 2024.* [[Paper](https://doi.org/10.1145/3678717.3691215)]

#### Distributed Storage Systems

1. **JuiceFS**: A high-performance cloud-native distributed file system designed for efficient storage and access of large-scale data. [[Github](https://github.com/juicedata/juicefs)]
2. **3FS**: A distributed file system designed for deep learning and large-scale data processing, emphasizing high throughput and reliability. [[Github](https://github.com/deepseek-ai/3fs)]
3. **S3**: A widely used cloud storage service offering secure, scalable, and highly available object storage solutions. [[Source](https://aws.amazon.com/s3)]
4. **Hdfs architecture guide. Hadoop apache project**  
    D. Borthakur et al. *Hadoop apache project, 53(1-13):2, 2008*. [[Source](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)]

#### Heterogeneous Storage Systems

1. **ProTrain: Efficient LLM Training via Memory-Aware Techniques**  
   Hanmei Yang, Jin Zhou, Yao Fu, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2406.08334)]
2. **ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning**  
   Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, et al. *SC 2021.* [[Paper](https://doi.org/10.1145/3458817.3476205)]
3. **ZeRO-Offload: Democratizing Billion-Scale Model Training**  
   Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, et al. *USENIX ATC 2021.* [[Paper](https://www.usenix.org/system/files/atc21-ren-jie.pdf)]
4. **ZeRO: memory optimizations toward training trillion parameter models**  
   Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, et al. *SC 2020.* [[Paper](https://dl.acm.org/doi/10.5555/3433701.3433727)]
5. **vDNN: virtualized deep neural networks for scalable, memory-efficient neural network design**  
   Minsoo Rhu, Natalia Gimelshein, Jason Clemons, et al. *MICRO-49 2016.* [[Paper](https://dl.acm.org/doi/10.5555/3195638.3195660)]



### 2.3 Data Organization

[⬆️top](#table-of-contents)

1. **Survey of Hallucination in Natural Language Generation**  
   Ziwei Ji, Nayeon Lee, Rita Frieske, et al. *ACM Computing Surveys (2022)*. [[Paper](https://dl.acm.org/doi/10.1145/3571730)]
2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
   Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. *NeurIPS 2020.* [[Paper](https://doi.org/10.48550/arXiv.2005.11401)]

#### Vector-Based Organization

1. **STELLA**: A large-scale Chinese vector database supporting efficient vector search and semantic retrieval applications. [[Source](https://huggingface.co/infgrad/stella-large-zh-v2)]
2. **Milvus**: An open-source vector database focused on large-scale, high-performance similarity search and analysis. [[Source](https://milvus.io)]
3. **Weaviate**: Weaviate offers a cloud-native vector search engine supporting intelligent search and knowledge graph construction for multimodal data. [[Source](https://weaviate.io)]
4. **LanceDB**: An efficient vector database designed for large-scale machine learning and recommendation systems. [[Source](https://lancedb.com)]
5. **Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation**  
   Zijie Zhong, Hanwen Liu, Xiaoya Cui, et al. *COLING 2025.* [[Paper](https://doi.org/10.48550/arXiv.2406.00456)]
6. **Dense X Retrieval: What Retrieval Granularity Should We Use?**  
   Tong Chen, Hongwei Wang, Sihao Chen, et al. *EMNLP 2024*. [[Paper](https://aclanthology.org/2024.emnlp-main.845/)]
7. **Scalable and Domain-General Abstractive Proposition Segmentation**  
   Mohammad Javad Hosseini, Yang Gao, Tim Baumgärtner, et al. *Findings of EMNLP 2024*. [[Paper](https://aclanthology.org/2024.findings-emnlp.517/)]
8. **A Hierarchical Context Augmentation Method to Improve Retrieval-Augmented LLMs on Scientific Papers**  
   Tian-Yi Che, Xian-Ling Mao, Tian Lan, et al. *KDD 2024*. [[Paper](https://dl.acm.org/doi/10.1145/3637528.3671847)]
9. **M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**  
   Jianlyu Chen, Shitao Xiao, Peitian Zhang, et al. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.137.pdf)]
10. **Thread: A Logic-Based Data Organization Paradigm for How-To Question Answering with Retrieval Augmented Generation**  
    Kaikai An, Fangkai Yang, Liqun Li, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2406.13372)]
11. **GleanVec: Accelerating Vector Search with Minimalist Nonlinear Dimensionality Reduction**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2410.22347)]
12. **The Faiss Library**  
    Matthijs Douze, Alexandr Guzhva, Chengqi Deng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.08281)]
13. **Similarity Search in the Blink of an Eye with Compressed Indices**  
    Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, et al. *VLDB Endowment 2023.* [[Paper](https://doi.org/10.14778/3611479.3611537)]
14. **LeanVec: Searching Vectors Faster by Making Them Fit**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, et al. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2312.16335)]
15. **Towards General Text Embeddings with Multi-stage Contrastive Learning**  
    Zehan Li, Xin Zhang, Yanzhao Zhang, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2308.03281)]

#### Graph-Based Organization

1. **ArangoDB**: A multi-model database that supports graph, document, and key-value data, suitable for handling complex relational queries. [[Source](https://arangodb.com/)]
2. **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**  
   Tianyu Fan, Jingyuan Wang, Xubin Ren, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.06713)]
3. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**  
   Darren Edge, Ha Trinh, Newman Cheng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.16130)]
4. **LightRAG: Simple and Fast Retrieval-Augmented Generation**  
   Zirui Guo, Lianghao Xia, Yanhua Yu, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.05779)]
5. **Graph Databases Assessment: JanusGraph, Neo4j, and TigerGraph**  
   Jéssica Monteiro, et al. *Perspectives and Trends in Education and Technology 2023.* [[Paper](https://doi.org/10.1007/978-981-19-6585-2_58)]
6. **Empirical Evaluation of a Cloud-Based Graph Database: the Case of Neptune**  
   Ghislain Auguste Atemezing. *KGSWC 2021.* [[Paper](https://doi.org/10.1007/978-3-030-91305-2_3)]



### 2.4 Data Movement

[⬆️top](#table-of-contents)

#### Caching Data

1. **CacheLib**: An open-source, high-performance embedded caching library developed by Meta to accelerate data access and increase system throughput. [[Source](https://cachelib.org/)]
2. **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**  
   Mark Zhao, Satadru Pan, Niket Agarwal, et al. *USENIX ATC 2023.* [[Paper](https://www.usenix.org/conference/atc23/presentation/zhao)]
3. **Fluid: Dataset Abstraction and Elastic Acceleration for Cloud-native Deep Learning Training Jobs**  
   Rong Gu, Kai Zhang, Zhihao Xu, et al. *ICDE 2022.* [[Paper](https://doi.org/10.1109/ICDE53745.2022.00209)]
4. **Quiver: An Informed Storage Cache for Deep Learning**  
   Abhishek Kumar, Muthian Sivathanu. *USENIX FAST 2020.* [[Paper](https://www.usenix.org/conference/fast20/presentation/kumar)]

#### Data/Operator Offloading

1. **cedar: Optimized and Unified Machine Learning Input Data Pipelines**  
   Mark Zhao, et al. *Proceedings of the VLDB Endowment, Volume 18, Issue 2, 2025.* [[Paper](https://dl.acm.org/doi/10.14778/3705829.3705861)]
2. **Pecan: cost-efficient ML data preprocessing with automatic transformation ordering and hybrid placement**  
   Dan Graur, Oto Mraz, Muyu Li, et al. *USENIX ATC 2024.* [[Paper](https://dl.acm.org/doi/10.5555/3691992.3692032)]
3. **tf.data service: A Case for Disaggregating ML Input Data Processing**  
   Andrew Audibert, Yang Chen, Dan Graur, et al. *SoCC 2023.* [[Paper](https://doi.org/10.1145/3620678.3624666)]
4. **Cachew: Machine Learning Input Data Processing as a Service**  
   Dan Graur, Damien Aymon, Dan Kluser, et al. *USENIX ATC 2022.* [[Paper](https://www.usenix.org/conference/atc22/presentation/graur)]
5. **Borg: the next generation**  
   Muhammad Tirmazi, Adam Barker, Nan Deng, et al. *EuroSys 2020*. [[Paper](https://dl.acm.org/doi/10.1145/3342195.3387517)]

#### Overlapping of storage and computing

1. **Optimizing RLHF Training for Large Language Models with Stage Fusion**  
   Yinmin Zhong, Zili Zhang, Bingyang Wu, et al. *NSDI 2025*. [[Paper](https://www.usenix.org/conference/nsdi25/presentation/zhong)]
2. **SiloD: A Co-design of Caching and Scheduling for Deep Learning Clusters**  
   Hanyu Zhao, Zhenhua Han, Zhi Yang, et al. *EuroSys 2023.* [[Paper](https://doi.org/10.1145/3552326.3567499)]
3. **Optimization by Simulated Annealing**  
   S. Kirkpatrick, C. D. Gelatt, Jr., M. P. Vecchi. *Science, 220(4598):671–680, 1983*. [[Paper](https://www.science.org/doi/10.1126/science.220.4598.671)]



### 2.5 Data Fault Tolerance

[⬆️top](#table-of-contents)

#### Checkpoints

1. **PaddleNLP**: PaddleNLP supports checkpoint saving and resuming during training, enabling fault tolerance and recovery for long-running training tasks. [[Source](https://paddlenlp.readthedocs.io)]
2. **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**  
   Ziheng Jiang, Haibin Lin, Yinmin Zhong, et al. *USENIX NSDI 2024.* [[Paper](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)]
3. **ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development**  
   Borui Wan, Mingji Han, Yiyao Sheng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2407.20143)]
4. **GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints**  
   Zhuang Wang, Zhen Jia, Shuai Zheng, et al. *SOSP 2023.* [[Paper](https://doi.org/10.1145/3600006.3613145)]
5. **CheckFreq: Frequent, Fine-Grained DNN Checkpointing**  
   Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *USENIX FAST 2021.* [[Paper](https://www.usenix.org/conference/fast21/presentation/mohan)]

#### Redundant Computations

1. **ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation**  
   Swapnil Gandhi, Mark Zhao, Athinagoras Skiadopoulos, et al. *SOSP 2024*. [[Paper](https://arxiv.org/abs/2405.14009)]
2. **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs**  
   John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, et al.  *NSDI 2023* . [[Paper](https://www.usenix.org/conference/nsdi23/presentation/thorpe)]
3. **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates**  
   Insu Jang, Zhenning Yang, Zhen Zhang, et al. *SOSP 2023*. [[Paper](https://dl.acm.org/doi/10.1145/3600006.3613152)]



### 2.6 KV Cache

[⬆️top](#table-of-contents)

#### Cache Space Management

1. **Efficient Memory Management for Large Language Model Serving with PagedAttention**  
   Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. *SOSP 2023.* [[Paper](https://arxiv.org/abs/2309.06180)]
2. **VTensor: Using Virtual Tensors to Build a Layout-oblivious AI Programming Framework**  
   Feng Yu, Jiacheng Zhao, Huimin Cui, et al. *PACT 2020.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3410463.3414664)]

#### KV Placement

1. **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**  
   Bin Gao, Zhuomin He, Puru Sharma, et al. *USENIX ATC 2024.* [[Paper](https://arxiv.org/abs/2403.19708)]
2. **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**  
   Chao Jin, Zili Zhang, Xuanlin Jiang, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2404.12457)]

#### KV Shrinking

1. **Adaptive KV-Cache Compression without Manually Setting Budget**     
   Chenxia Tang, Jianchun Liu, Hongli Xu, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2509.03136)]
2. **Fast State Restoration in LLM Serving with HCache**  
   Shiwei Gao, Youmin Chen, Jiwu Shu. *EuroSys 2025.* [[Paper](https://arxiv.org/abs/2410.05004)]
3. **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**  
   Yuhan Liu, Hanchen Li, Yihua Cheng, et al. *SIGCOMM 2024.* [[Paper](https://dl.acm.org/doi/abs/10.1145/3651890.3672274)]
4. **MiniCache: KV Cache Compression in Depth Dimension for Large Language Models**  
   Akide Liu, Jing Liu, Zizheng Pan, et al. *NeurIPS 2024*. [[Paper](https://neurips.cc/virtual/2024/poster/93380)]
5. **Animating rotation with quaternion curves**  
   Ken Shoemake. *ACM SIGGRAPH Computer Graphics, Volume 19, Issue 3. 1985*. [[Paper](https://dl.acm.org/doi/10.1145/325165.325242)]

#### KV Indexing

1. **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**  
   Lu Ye, Ze Tao, Yong Huang, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.623/)]
2. **BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching**  
   Zhen Zheng, Xin Ji, Taosong Fang, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.03594)]



## 3 Data Serving for LLM

[⬆️top](#table-of-contents)

### 3.1 Data Shuffling

#### Data Shuffling for Training

1. **Mixtera: A Data Plane for Foundation Model Training**      
   Maximilian Böther, Xiaozhe Yao, Tolga Kerimoglu, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2502.19790)]
2. **Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training**  
   Zheheng Luo, Xin Zhang, Xiao Liu, et al. *ACL 2025*. [[Paper](https://arxiv.org/abs/2411.14318)]
3. **How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition**  
   Guanting Dong, Hongyi Yuan, Keming Lu, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.12/)]
4. **Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models**    
   Minghao Wu, Thuy-Trang Vu, Lizhen Qu, et al. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.787/)]
5. **Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning**  
   Jisu Kim, Juhwan Lee. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.07490)]
6. **NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks**  
   Jean-michel Attendu, Jean-philippe Corbeil. *SustaiNLP @ ACL 2023.* [[Paper](https://aclanthology.org/2023.sustainlp-1.9/)]
7. **Efficient Online Data Mixing For Language Model Pre-Training**  
   Alon Albalak, Liangming Pan, Colin Raffel, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2312.02406)]
8. **Data Pruning via Moving-one-Sample-out**  
   Haoru Tan, Sitong Wu, Fei Du, et al. *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2310.14664)]
9. **BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning**  
   Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi, et al. *ENLSP @ NeurIPS2022.* [[Paper](https://doi.org/10.48550/arXiv.2211.05610)]
10. **Scaling Laws for Neural Language Models**  
   Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *arXiv 2020*. [[Paper](https://arxiv.org/abs/2001.08361)]
11. **Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory**  
    James L. McClelland, Bruce L. McNaughton, Randall C. O’Reilly. *Psychological Review 1995.* [[Paper](https://cseweb.ucsd.edu/~gary/258/jay.pdf)]
12. **Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem**  
    M. McCloskey, N. J. Cohen. *Psychology of Learning and Motivation 1989.* [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)]



#### Data Selection for RAG

1. **Cohere rerank**: Cohere's rerank model reorders initial retrieval results to improve relevance to the query, making it a key component for building high-quality RAG systems. [[Source](https://docs.cohere.com)]
2. **ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval**  
   Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, et al. *NAACL 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.15245)]
3. **MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation**  
   Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2501.00332)]
4. **ARAGOG: Advanced RAG Output Grading**  
   Matouš Eibich, Shivay Nagpal, Alexander Fred-Ojala. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.01037)]
5. **Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!**  
   Yubo Ma, Yixin Cao, YongChing Hong, et al. *Findings of EMNLP 2023*. [[Paper](https://aclanthology.org/2023.findings-emnlp.710/)]
6. **Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model**  
   Jiaxi Cui, Munan Ning, Zongjian Li, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2306.16092v2)]
7. **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**  
   Ronak Pradeep, Sahel Sharifymoghaddam, Jimmy Lin. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2309.15088)]



### 3.2 Data Compression

[⬆️top](#table-of-contents)

#### RAG Knowledge Compression

1. **Context Embeddings for Efficient Answer Generation in RAG**  
   David Rau, Shuai Wang, Hervé Déjean, et al. *WSDM 2025.* [[Paper](https://doi.org/10.48550/arXiv.2407.09252)]
2. **xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**  
   Xin Cheng, Xun Wang, Xingxing Zhang, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.13792)]
3. **RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation**  
   Fangyuan Xu, Weijia Shi, Eunsol Choi. *ICLR 2024.* [[Paper](https://iclr.cc/virtual/2024/poster/17885)]
4. **Compressing Long Context for Enhancing RAG with AMR-based Concept Distillation**   
   Kaize Shi, Xueyao Sun, Qing Li, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.03085)]
5. **Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation**  
   Dongwon Jung, Qin Liu, Tenghao Huang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.12468)]

#### Prompt Compression

1. **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**  
   Huiqiang Jiang, Qianhui Wu, Xufang Luo, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.91/)]
2. **LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression**  
   Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, et al. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.57/)]
3. **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**  
   Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, et al. *EMNLP 2023.* [[Paper](https://aclanthology.org/2023.emnlp-main.825.pdf)]
4. **Learning to Compress Prompts with Gist Tokens**  
   Jesse Mu, Xiang Lisa Li, Noah Goodman. *NeurIPS 2023.* [[Paper](https://arxiv.org/abs/2304.08467)]
5. **Adapting Language Models to Compress Contexts**  
   Alexis Chevalier, Alexander Wettig, Anirudh Ajith, et al. *EMNLP 2023.* [[Paper](https://aclanthology.org/2023.emnlp-main.232.pdf)]



### 3.3 Data Packing

[⬆️top](#table-of-contents)

#### Short Sequence Insertion

1. **Fewer Truncations Improve Language Modeling**  
   Hantian Ding, Zijian Wang, Giovanni Paolini, et al. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.10830)]
2. **Bucket Pre-training is All You Need**  
   Hongtao Liu, Qiyao Peng, Qing Yang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.07495)]

#### Sequence Combination Optimization

1. **Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum**  
   Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.13226)]
2. **Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance**  
   Mario Michael Krell, Matej Kosec, Sergio P. Perez, et al. *arXiv 2021.* [[Paper](https://doi.org/10.48550/arXiv.2107.02027)]

#### Semantic-Based Packing

1. **Structured Packing in LLM Training Improves Long Context Utilization**  
   Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, et al. *AAAI 2025.* [[Paper](https://doi.org/10.48550/arXiv.2312.17296)]
2. **In-context Pretraining: Language Modeling Beyond Document Boundaries**  
   Weijia Shi, Sewon Min, Maria Lomeli, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2310.10638)]



### 3.4 Data Provenance

[⬆️top](#table-of-contents)

1. **A comprehensive survey on data provenance: : State-of-the-art approaches and their deployments for IoT security enforcement**   
   Md Morshed Alam, Weichao Wang. *Journal of Computer Security, Volume 29, Issue 4. 2021*. [[Paper](https://dl.acm.org/doi/abs/10.3233/JCS-200108)]

#### Embedding Markers

1. **Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**  
   Tong Zhou, Xuandong Zhao, Xiaolin Xu, et al. *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2406.01946)]
2. **Undetectable Watermarks for Language Models**  
   Miranda Christ, et al. in *Proceedings of the 37th Annual Conference on Learning Theory (COLT 2024)*. [[Paper](https://arxiv.org/abs/2306.09194)]
3. **An Unforgeable Publicly Verifiable Watermark for Large Language Models**  
   Aiwei Liu, Leyi Pan, Xuming Hu, et al. *ICLR 2024*. [[Paper](https://arxiv.org/abs/2307.16230)]
4. **A Watermark for Large Language Models**  
   John Kirchenbauer, Jonas Geiping, Yuxin Wen, et al. *ICML 2023*. [[Paper](https://arxiv.org/abs/2301.10226)]
5. **Publicly-Detectable Watermarking for Language Models**   
   Jaiden Fairoze, Sanjam Garg, Somesh Jha, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2310.18491)]

#### Statistical Provenance

1. **A Watermark for Large Language Models** [[Paper](https://arxiv.org/abs/2301.10226)]



## 4 LLM for Data Management

[⬆️top](#table-of-contents)

### 4.1 LLM for Data Manipulation

#### 4.1.1 LLM for Data Cleaning

##### Data Standardization

1. **Exploring the Feasibility of Automated Data Standardization using Large Language Models for Seamless Positioning**

   Lee, Max JL, et al.  *2024 14th International Conference on Indoor Positioning and Indoor Navigation (IPIN)*. IEEE, 2024. [[Paper](https://arxiv.org/abs/2408.12080)]

2. **Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes**  
   Simran Arora, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 2, 2024.* [[Paper](https://dl.acm.org/doi/abs/10.14778/3626292.3626294)]

3. **CleanAgent: Automating Data Standardization with LLM-based Agents**  
   Danrui Qi, Jiannan Wang. *arXiv 2024.* [[Paper](https://arxiv.org/pdf/2403.08291)]

4. **AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark**  
   Lan Li, Liri Fang, Vetle I. Torvik. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.06724)]

5. **LLMs with User-defined Prompts as Generic Data Operators for Reliable Data Processing**  
   Luyi Ma, et al. *1st IEEE International Workshop on Data Engineering and Modeling for AI (DEMAI), IEEE BigData 2023.* [[Paper](https://arxiv.org/abs/2312.16351)]

6. **Large language models as data preprocessors.**  

   Zhang, Haochen, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2308.16361)]

##### Data Error Processing

1. **Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets**  
   Tommaso Bendinelli, Artur Dox, Christian Holz. *ICLR 2025 Workshop on Foundation Models in the Wild*. [[Paper](https://arxiv.org/abs/2503.06664)]

2. **Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets**

   Bendinelli, Tommaso, Artur Dox, and Christian Holz. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.06664)]

3. **ZeroED: Hybrid Zero-shot Error Detection through Large Language Model Reasoning**

   Ni, Wei, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2504.05345v1)]

4. **GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models**  
   Mengyi Yan, et al. *Proceedings of the ACM on Management of Data, Volume 2, Issue 6, 2024.* [[Paper](https://dl.acm.org/doi/10.1145/3698811)]

5. **Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation**  
   Juhwan Choi, Jungmin Yun, Kyohoon Jin, et al. *EMNLP 2024*. [[Paper](https://arxiv.org/abs/2404.09682)]

6. **Data Cleaning Using Large Language Models**  
   Shuo Zhang, Zezhou Huang, Eugene Wu. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.15547)]

7. **LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs**  
   Fabian Biester, Mohamed Abdelaal, Daniel Del Gaudio. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2404.18681)]

8. **Anomaly Detection of Tabular Data Using LLMs**

   Li, Aodong, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2406.16308)]

9. **Cleaning Semi-Structured Errors in Open Data Using Large Language Models**

   M. Mondal, J. Audiffren, L. Dolamic, et al, *2024 11th IEEE Swiss Conference on Data Science (SDS)*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10675754)]

10. **IterClean: An Iterative Data Cleaning Framework with Large Language Models**       
    Wei Ni, et al. *Proceedings of the ACM Turing Award Celebration Conference - China 2024*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3674399.3674436)]

##### Data Imputation

1. **Does Prompt Design Impact Quality of Data Imputation by LLMs?**

   Srinivasan, Shreenidhi, and Lydia Manikonda. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2506.04172)]

2. **On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing**

   Wang, Jianwei, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2501.02191)]

3. **RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes**  
   Zan Ahmad Naeem, et al. *VLDB Endowment 2024*. [[Paper](https://dl.acm.org/doi/10.14778/3685800.3685890)]

4. **Data Augmentation using Large Language Models: Data Perspectives, Learning Paradigms and Challenges**

   Ding, Bosheng, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2403.02990)]

5. **A Context-Aware Approach for Enhancing Data Imputation with Pre-trained Language Models**

   Hayat, Ahatsham, and Mohammad Rashedul Hasan. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2405.17712v2)]

#### 4.1.2 LLM for Data Integration

##### Entity Matching

1. **A Deep Dive Into Cross-Dataset Entity Matching with Large and Small Language Models**

   Zhang, Zeyu, et al. *International Conference on Extending Database Technology (EDBT) 2025.* [[Paper](https://deem.berlin/publication/2025-02-05-a-deep-dive-into-cross-dataset-em-with-small-and-large-language-models-edbt/)]

2. **Large Language Models for Data Discovery and Integration: Challenges and Opportunities**

   Freire, Juliana, et al.   *[IEEE Data Eng. Bull. 49(1)](https://dblp.org/db/journals/debu/debu49.html#FreireFFKLPSSW25): 3-31 (2025)*. [[Paper](http://sites.computer.org/debull/A25mar/p3.pdf)]

3. **Entity matching using large language models**  
   Ralph Peeters, Christian Bizer. *EDBT 2025.* [[Paper](https://arxiv.org/abs/2310.11244)]

4. **Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching**  
   Tianshu Wang, Hongyu Lin, Xiaoyang Chen, et al. *COLING 2025.* [[Paper](https://aclanthology.org/2025.coling-main.8/)]

5. **Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration**  
   Meihao Fan, Xiaoyue Han, Ju Fan, et al. *ICDE 2024.* [[Paper](https://ieeexplore.ieee.org/document/10597751)]

6. **KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs**  
   Yongqin Xu, Huan Li, Ke Chen, Lidan Shou. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.12480)]

7. **Jellyfish: A Large Language Model for Data Preprocessing**  
   Haochen Zhang, Yuyang Dong, Chuan Xiao, et al. *EMNLP 2024.* [[Paper](https://arxiv.org/abs/2312.01678)]

8. **Fine-tuning Large Language Models for Entity Matching**

   Steiner, Aaron, Ralph Peeters, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.08185)]

##### Schema Matching

1. **SCHEMORA: Schema Matching via Multi-stage Recommendation and Metadata Enrichment using Off-the-Shelf LLMs**    
   Osman Erman Gungor, Derak Paulsen, William Kang. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2507.14376v1)]

2. **Towards Scalable Schema Mapping using Large Language Models**

   Buss, Christopher, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.24716)]

3. **Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching**  
   Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2501.08686)]

4. **Interactive Data Harmonization with LLM Agents**  
   Aécio Santos, Eduardo H. M. Pena, Roque Lopez, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.07132)]

5. **Schema Matching with Large Language Models: an Experimental Study**  
   Marcel Parciak, Brecht Vandevoort, Frank Neven, et al. *TaDA 2024 Workshop, collocated with VLDB 2024.* [[Paper](https://doi.org/10.48550/arXiv.2407.11852)]

6. **Magneto: Combining Small and Large Language Models for Schema Matching**  
   Yurong Liu, Eduardo Pena, Aecio Santos, et al. *VLDB Endowment 2024.*  [[Paper](https://www.vldb.org/pvldb/vol17/p2750-fan.pdf)]

7. **Agent-OM: Leveraging LLM Agents for Ontology Matching**
   Zhangcheng Qiang, et al. *Proceedings of the VLDB Endowment, Volume 18, Issue 3, 2024.* [[Paper](https://dl.acm.org/doi/10.14778/3712221.3712222)]

8. **Matchmaker: Self-Improving Large Language Model Programs for Schema Matching**

   Seedat, Nabeel, and Mihaela van der Schaar. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.24105)]

9. **TableGPT2: A Large Multimodal Model with Tabular Data Integration**

   Su, Aofeng, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2411.02059)]

#### 4.1.3 LLM for Data Discovery

1. **ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models**  
   Benjamin Feuer, Yurong Liu, Chinmay Hegde, et al. *VLDB 2024*. [[Paper](https://arxiv.org/abs/2310.18208#:~:text=We%20introduce%20ArcheType%2C%20a%20simple%2C%20practical%20method%20for,solve%20CTA%20problems%20in%20a%20fully%20zero-shot%20manner.)]

##### Data Profiling

1. **Flexible metadata harvesting for ecology using large language models**    
   Zehao Lu, Thijs L van der Plas, Parinaz Rashidi, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2508.20115)]

2. **Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System**  
   Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, et al. *SIGMOD 2025*. [[Paper](https://arxiv.org/abs/2504.09207#:~:text=In%20this%20paper%2C%20we%20introduce%20Pneuma%2C%20a%20retrieval-augmented,designed%20to%20efficiently%20and%20effectively%20discover%20tabular%20data.)]

3. **AutoDDG: Automated Dataset Description Generation using Large Language Models**  
   Haoxiang Zhang, Yurong Liu, Wei-Lun (Allen) Hung, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2502.01050)]

4. **LEDD: Large Language Model-Empowered Data Discovery in Data Lakes**  
   Qi An, Chihua Ying, Yuqing Zhu, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.15182)]

5. **LLM-Aided Customizable Profiling of Code Data Based On Programming Language Concepts**

   Thorat, Pankaj, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2503.15571)]

6. **Cocoon: Semantic Table Profiling Using Large Language Models**

   Huang, Zezhou, et al. *Proceedings of the 2024 Workshop on Human-In-the-Loop Data Analytics*. 2024. [[Paper](https://dl.acm.org/doi/abs/10.1145/3665939.3665957)]

##### Data Annotation

1. **LLMs as Data Annotators: How Close Are We to Human Performance**     
   Haq, Muhammad Uzair Ul, Davide Rigoni, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2504.15022)]

2. **Birdie: Natural Language-Driven Table Discovery Using Differentiable Search Index**  
   Yuxiang Guo, Zhonghao Hu, Yuren Mao, et al. *VLDB 2025*. [[Paper](https://arxiv.org/abs/2504.21282)]

3. **Mind the Data Gap: Bridging LLMs to Enterprise Data Integration**  
   Moe Kayali, Fabian Wenz, Nesime Tatbul, et al. *CIDR 2025.* [[Paper](https://arxiv.org/abs/2412.20331)]

4. **Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation**  
   Keti Korini, Christian Bizer. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.02718)]

4. **Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models**    
   Ting Cai, Stephen Sheen, AnHai Doan. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2508.09403v2)]

4. **An LLM Agent-Based Complex Semantic Table Annotation Approach**    
   Yilin Geng, Shujing Wang, Chuan Wang, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2508.12868)]

5. **Open-Source LLMs for Text Annotation: A Practical Guide for Model Setting and Fine-Tuning**

   Alizadeh, Meysam, et al. *Journal of Computational Social Science* 8.1 (2025): 1-25. [[Paper](https://arxiv.org/abs/2307.02179)]

6. **Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation**

   Xia, Mingxuan, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2506.03857)]

7. **Evaluating how LLM annotations represent diverse views on contentious topics**

   Brown, Megan A., et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.23243)]

8. **CHORUS: Foundation Models for Unified Data Discovery and Exploration**  
   Moe Kayali, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[Paper](https://dl.acm.org/doi/10.14778/3659437.3659461)]

9. **RACOON: An LLM-based Framework for Retrieval-Augmented Column Type Annotation with a Knowledge Graph**  
   Lindsey Linxi Wei, Guorui Xiao, Magdalena Balazinska. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.14556)]

12. **AutoLabel: Automated Textual Data Annotation Method Based on Active Learning and Large Language Model**    
    Ming, Xuran, et al. *International Conference on Knowledge Science, Engineering and Management*. 2024. [[Paper](https://dl.acm.org/doi/10.1007/978-981-97-5501-1_30)]

11. **The Promises and Pitfalls of LLM Annotations in Dataset Labeling: a Case Study on Media Bias Detection**

    Horych, Tomas, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2411.11081)]

12. **Large Language Models as Annotators: Enhancing Generalization of NLP Models at Minimal Cost**

    Bansal, Parikshit, and Amit Sharma. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2306.15766)]
<!-- 
### 4.2 LLM for Data Analysis

[⬆️top](#table-of-contents)

#### 4.2.1 LLM for Structured Data Analysis

##### 4.2.1.1 Relational Data Analysis

###### LLM for Natural Language Interfaces

1. **Text to Query Plans for Question Answering on Large Tables**    
   Yipeng Zhang, Chen Wang, Yuzhe Zhang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2508.18758v1)]
2. **RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System**    
   Zui Chen, Han Li, Xinhao Zhang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2508.17590)]
3. **An advanced AI driven database system**     
   M. Tedeschi, S. Rizwan, C. Shringi, et al. *EDULEARN25 Conference Proceedings. 2025.* [[Paper](https://arxiv.org/abs/2507.17778)]
4. **Cracking SQL Barriers: An LLM-based Dialect Translation System**  
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *SIGMOD 2025*. [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/SIGMOD25-CrackSQL.pdf)]
5. **CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models**    
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2504.00882#:~:text=In%20this%20demonstration%2C%20we%20present%20CrackSQL%2C%20the%20first,rule%20and%20LLM-based%20methods%20to%20overcome%20these%20limitations.)]
6. **Automatic Metadata Extraction for Text-to-SQL**     
   Vladislav Shkapenyuk, Divesh Srivastava, Theodore Johnson, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.19988)]
7. **CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning**     
   Lei Sheng, Shuai-Shuai Xu. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.13271)]
8. **Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL**       
   Lei Sheng, Shuai-Shuai Xu. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.13271)]
9. **OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale**    
   Haoyang Li, Shang Wu, Xiaokang Zhang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2503.02240)]
10. **OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment**   
   Xiangjin Xie, Guangwei Xu, Lingyan Zhao, Ruijie Guo. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2502.14913)]
11. **Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning**    
    Yusuf Denizay Dönder, Derek Hommel, Andrea W Wen-Yi, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.14174)]
12. **A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL**   
    Yingqi Gao, Yifu Liu, Xiaoxia Li, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2411.08599)]
13. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis**  
    Chao Zhang, Yuren Mao, Yijiang Fan, et al. *SIGMOD 2024.* [[Paper](https://doi.org/10.1145/3626246.3653375)]
14. **CodeS: Towards Building Open-source Language Models for Text-to-SQL**  
    Haoyang Li, et al. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024.* [[Paper](https://doi.org/10.1145/3654930)]
15. **The Dawn of Natural Language to SQL: Are We Fully Ready?**  
    Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, Nan Tang. *VLDB 2024.* [[Paper](https://arxiv.org/abs/2406.01265)]
16. **Contextualized Data-Wrangling Code Generation in Computational Notebooks**  
    Junjie Huang, Daya Guo, Chenglong Wang, et al. *ASE 2024*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3691620.3695503)]
17. **PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**  
    Zhishuai Li, Xiang Wang, Jingjing Zhao, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.09732)]
18. **CHESS: Contextual Harnessing for Efficient SQL Synthesis**   
    Shayan Talaei, Mohammadreza Pourreza, Yu-Chen Chang, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.16755)]
19. **Data Interpreter: An LLM Agent For Data Science**  
    Sirui Hong, Yizhang Lin, Bang Liu, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2402.18679)]
20. **DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction**  
    Mohammadreza Pourreza, Davood Rafiei. *NeurIPS 2023*. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667699)]
21. **Natural Language to Code Generation in Interactive Data Science Notebooks**   
    Pengcheng Yin, Wen-Ding Li, Kefan Xiao, et al. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-long.9/)]
22. **PaLM: Scaling Language Modeling with Pathways**   
    Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. *JMLR 2023.* [[Paper](https://dl.acm.org/doi/10.5555/3648699.3648939)]

###### LLM for Semantic Analysis

1. **TableMaster: A Recipe to Advance Table Understanding with Language Models**  
   Lang Cao. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.19378)]
2. **RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals**  
   Xuanliang Zhang, Dingzirui Wang, Keyan Xu, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.15110)]
3. **PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models**  
   Wei Zhou, Mohsen Mesgar, Heike Adel, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.17565)]
4. **TAT-LLM: A Specialized Language Model for Discrete Reasoning over Financial Tabular and Textual Data**  
   Fengbin Zhu, Ziyang Liu, Fuli Feng, et al. *ICAIF 2024.* [[Paper](https://doi.org/10.1145/3677052.3698685)]
5. **CABINET: Content Relevance based Noise Reduction for Table Question Answering**  
   Sohan Patnaik, Heril Changwal, Milan Aggarwal, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.01155)]
6. **Multimodal Table Understanding**  
   Mingyu Zheng, Xinwei Feng, Qingyi Si, et al. *ACL 2024*. [[Paper](https://aclanthology.org/2024.acl-long.493/)]
7. **TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy**  
   Weichao Zhao, Hao Feng, Qi Liu, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.01326)]
8. **TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning**  
   Yilun Zhao, Lyuhao Chen, Arman Cohan, Chen Zhao. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.692/)]
9. **ReAcTable: Enhancing ReAct for Table Question Answering**  
   Yunjia Zhang, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[Paper](https://doi.org/10.14778/3659437.3659452)]
10. **Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**  
    Zilong Wang, Hao Zhang, Chun-Liang Li, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.04398)]
11. **Table-GPT: Table Fine-tuned GPT for Diverse Table Tasks**  
    Peng Li, et al. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024*. [[Paper](https://dl.acm.org/doi/10.1145/3654979)]
12. **TableGPT2: A Large Multimodal Model with Tabular Data Integration**  
    Aofeng Su, Aowen Wang, Chao Ye, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2411.02059)]
13. **S3HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering**   
    Fangyu Lei, Xiang Li, Yifan Wei, et al. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-short.147/)]
14. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** [[Paper](https://doi.org/10.48550/arXiv.2306.05685)]



##### 4.2.1.2 Graph Data Analysis

1. **Blazegraph**: A high-performance graph database that supports RDF/SPARQL queries, commonly used in semantic web and knowledge graph analysis. [[Source](https://blazegraph.com/)]
2. **GraphDB**: A triplestore with ontology reasoning and SPARQL query support, widely used for enterprise knowledge management and semantic search. [[Source](https://graphdb.ontotext.com/)]
3. **Neo4j**: Neo4j is one of the most popular graph databases, based on the property graph model, supporting complex relationship queries and visual analytics. [[Github](https://github.com/neo4j/neo4j)]
4. **A Comparison of Current Graph Database Models**   
   Renzo Angles. *ICDEW 2012.* [[Paper](https://doi.org/10.1109/ICDEW.2012.31)]

###### Natural Language To Graph Analysis Query

1. **R3-NL2GQL: A Model Coordination and Knowledge Graph Alignment Approach for NL2GQL**   
   Yuhang Zhou, Yu He, Siyu Tian, et al. *Findings of EMNLP 2024.* [[Paper](https://aclanthology.org/2024.findings-emnlp.800/)]
2. **NAT-NL2GQL: A Novel Multi-Agent Framework for Translating Natural Language to Graph Query Language**  
   Yuanyuan Liang, Tingyu Xie, Gan Peng, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.10434)]
3. **Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks**  
   Xunkai Li, Zhengyu Wu, Jiayi Wu, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.12456)]
4. **Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey**  
   Qizhi Pei, Lijun Wu, Kaiyuan Gao, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2403.01528)]

###### LLM-based Semantic Analysis

1. **GraphGPT: Graph Instruction Tuning for Large Language Models**   
   Jiabin Tang, Yuhao Yang, Wei Wei, et al. *SIGIR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2310.13023)]
2. **Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models**   
   Guanming Xiong, Junwei Bao, Wen Zhao. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.569/)]
3. **FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering**     
   Zhenyu Li, Sunqi Fan, Yu Gu, et al. *AAAI 2024.* [[Paper](https://doi.org/10.48550/arXiv.2308.12060)]
4. **Language is All a Graph Needs**   
   Ruosong Ye, Caiqi Zhang, Runhui Wang, et al. *EACL 2024.* [[Paper](https://aclanthology.org/2024.findings-eacl.132/)]
5. **InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment**   
   Jianing Wang, Junda Wu, Yupeng Hou, et al. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.801/)]
6. **Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments**  
   Sitao Cheng, Ziyuan Zhuang, Yong Xu, et al. *Findings of ACL 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.08593)]
7. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**   
   Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. *NeurIPS 2023.* [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)]
8. **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**   
   Jinhao Jiang, Kun Zhou, Zican Dong, et al. *EMNLP 2023.* [[Paper](https://doi.org/10.48550/arXiv.2305.09645)]
9. **UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph**   
   Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, et al. *ICLR 2023.* [[Paper](https://doi.org/10.48550/arXiv.2212.00959)]
10. **Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering**   
    Jing Zhang, Xiaokang Zhang, Jifan Yu, et al. *ACL 2022.* [[Paper](https://aclanthology.org/2022.acl-long.396/)]
11. **RoBERTa: A Robustly Optimized BERT Pretraining Approach**  
    Yinhan Liu, Myle Ott, Naman Goyal, et al. *arXiv 2019*. [[Paper](https://arxiv.org/abs/1907.11692)]
12. **Inductive representation learning on large graphs**   
    William L. Hamilton, Rex Ying, Jure Leskovec. *NeurIPS 2017.* [[Paper](https://dl.acm.org/doi/10.5555/3294771.3294869)]
13. **Semi-Supervised Classification with Graph Convolutional Networks**   
    Thomas N. Kipf, Max Welling. *ICLR 2017.* [[Paper](https://doi.org/10.48550/arXiv.1609.02907)]



#### 4.2.2 LLM for Semi-Structured Data Analysis

1. **ST-Raptor: LLM-Powered Semi-Structured Table Question Answering**    
   Zirui Tang, Boyu Niu, Xuanhe Zhou, et al. *SIGMOD 2026.* (2025). [[Paper](https://arxiv.org/abs/2508.18190v3)]
2. **Querying Semi-Structured Data**  
   Serge Abiteboul. *ICDT 1997.* [[Paper](https://dl.acm.org/doi/10.5555/645502.656103)]

##### 4.2.2.1 Markup Language

##### 4.2.2.2 Semi-Structured Tables

1. **MiMoTable: A Multi-scale Spreadsheet Benchmark with Meta Operations for Table Reasoning**  
   Zheng Li, Yang Du, Mao Zheng, et al. *COLING 2025.* [[Paper](https://doi.org/10.48550/arXiv.2412.11711)]
2. **AOP: Automated and Interactive LLM Pipeline Orchestration for Answering Complex Queries**  
   Jiayi Wang, Guoliang Li. *CIDR 2025* [[Paper](https://vldb.org/cidrdb/papers/2025/p32-wang.pdf)]
3. **SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation**  
   Zeyao Ma, Bohan Zhang, Jing Zhang, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.14991)]
4. **TempTabQA: Temporal Question Answering for Semi-Structured Tables**  
   Vivek Gupta, Pranshu Kandoi, Mahek Bhavesh Vora, et al. *EMNLP 2023.* [[Paper](https://doi.org/10.48550/arXiv.2311.08002)]



#### 4.2.3 LLM for Unstructured Data Analysis

##### 4.2.3.1 Documents

1. **AOP: Automated and Interactive LLM Pipeline Orchestration for Answering Complex Queries** [[Paper](https://vldb.org/cidrdb/papers/2025/p32-wang.pdf)]
2. **Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing**  
 Chunwei Liu, Matthew Russo, Michael Cafarella, et al. *CIDR 2025* [[Paper](https://www.vldb.org/cidrdb/papers/2025/p12-liu.pdf)]
3. **Towards Accurate and Efficient Document Analytics with Large Language Models**  
 Y. Lin, M. Hulsebos, R. Ma, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.04674/)]
4. **DocFormerv2: Local Features for Document Understanding**  
   Srikar Appalaraju, Peng Tang, Qi Dong, et al. *AAAI 2024.* [[Paper](https://doi.org/10.1609/aaai.v38i2.27828)]
5. **mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding**  
   Anwen Hu, Haiyang Xu, Jiabo Ye, et al. *Findings of EMNLP 2024.* [[Paper](https://aclanthology.org/2024.findings-emnlp.175/)]
6. **DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding**  
   Hao Feng, Qi Liu, Hao Liu, et al. *SCIS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2311.11810)]
7. **Focus Anywhere for Fine-grained Multi-page Document Understanding** [[Paper](https://arxiv.org/abs/2405.14295)]
8. **General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model** [[Paper](https://arxiv.org/abs/2409.01704v1)]
9. **DUBLIN: Visual Document Understanding By Language-Image Network**  
   Kriti Aggarwal, Aditi Khandelwal, Kumar Tanmay, et al. *EMNLP Industry Track 2023.* [[Paper](https://aclanthology.org/2023.emnlp-industry.65/)]
10. **Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding**  
    Kenton Lee, Mandar Joshi, Iulia Turc, et al. *ICML 2023.* [[Paper](https://dl.acm.org/doi/10.5555/3618408.3619188?ref=localhost)]
11. **Unifying Vision, Text, and Layout for Universal Document Processing**  
    Zineng Tang, Ziyi Yang, Guoxin Wang, et al. *CVPR 2023.* [[Paper](https://arxiv.org/abs/2212.02623v3)]
12. **Exploring the limits of transfer learning with a unified text-to-text transformer** [[Paper](https://arxiv.org/abs/1910.10683v4)]
13. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
    Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. *ICLR  2021.* [[Paper](https://iclr.cc/virtual/2021/oral/3458)]
14. **The JPEG Still Picture Compression Standard**  
    Gregory K. Wallace. *Communications of the ACM 1991.* [[Paper](https://doi.org/10.1145/103085.103089)]



##### 4.2.3.2 Program Language Analysis

###### LLM as Program Vulnerability Detection Tools

1. **Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks**  
   Zhongxin Liu, Zhijie Tang, Junwei Zhang, et al. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3639142)]
2. **Large Language Model for Vulnerability Detection: Emerging Results and Future Directions**  
   Xin Zhou, Ting Zhang, David Lo. *ICSE-NIER 2024.* [[Paper](https://doi.org/10.1145/3639476.3639762)]
3. **Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code**  
   Junwei Zhang, Zhongxin Liu, Xing Hu, et al. *IEEE TSE 2023.* [[Paper](https://ieeexplore.ieee.org/document/10153647)]
4. **Software Vulnerability Detection with GPT and In-Context Learning**  
   Zhihong Liu, Qing Liao, Wenchao Gu, et al. *DSC 2023.* [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
5. **CodeBERT: A Pre-Trained Model for Programming and Natural Languages**  
   Zhangyin Feng, Daya Guo, Duyu Tang, et al. *Findings of EMNLP 2020.* [[Paper](https://aclanthology.org/2020.findings-emnlp.139/)]
6. **The Probabilistic Relevance Framework: BM25 and Beyond**  
   Stephen Robertson, et al. *Foundations and Trends in Information Retrieval, Volume 3, Issue 4, 2009.* [[Paper](https://dl.acm.org/doi/10.1561/1500000019)]

###### LLM-based Semantic-aware Analysis

1.  **Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning**  
   Mingyang Geng, Shangwen Wang, Dezun Dong, et al. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3608134)]
2.  **Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)**  
   Toufique Ahmed, Kunal Suresh Pai, Premkumar Devanbu, Earl Barr. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3639183)]
3.  **CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context**  
   Yangruibo Ding, Zijian Wang, Wasi Ahmad, et al. *LREC-COLING 2024.* [[Paper](https://aclanthology.org/2024.lrec-main.305/)]
4.  **Repoformer: Selective Retrieval for Repository-Level Code Completion**  
   Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, et al. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.10059)]
5.  **SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation**  
   Yingjie Mao, Xiaoqi Li, Wenkai Li, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.04863)]
6.  **Code Structure–Guided Transformer for Source Code Summarization**  
   Shuzheng Gao, et al. *ACM Transactions on Software Engineering and Methodology 2023.* [[Paper](https://doi.org/10.1145/3522674)]
7.  **RepoFusion: Training Code Models to Understand Your Repository**  
   Disha Shrivastava, Denis Kocetkov, Harm de Vries, et al. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2306.10998)]
 -->


### 4.2 LLM for Data System Optimization

[⬆️top](#table-of-contents)

#### 4.2.1 LLM for Configuration Tuning

1. **ELMo-Tune-V2: LLM-Assisted Full-Cycle Auto-Tuning to Optimize LSM-Based Key-Value Stores**  
    Viraj Thakkar, Qi Lin, Kenanya Keandra Adriel Prasetyo, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.17606)]
2. **MLETune: Streamlining Database Knob Tuning via Multi-LLMs Experts Guided Deep Reinforcement Learning**    
    Wenlong Dong, Wei Liu, Rui Xi, et al. *ICPADS 2024.* [[Paper](https://doi.org/10.1109/ICPADS63350.2024.00038)]

##### Tuning Task-Aware Prompt Engineering

1. **λ-Tune: Harnessing Large Language Models for Automated Database System Tuning**  
   Victor Giannankouris, Immanuel Trummer. *SIGMOD 2025.* [[Paper](https://doi.org/10.48550/arXiv.2411.03500)]
2. **LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model**  
   Xinxin Zhao, Haoyang Li, Jing Zhang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2503.07884)]
3. **LATuner: An LLM-Enhanced Database Tuning System Based on Adaptive Surrogate Model**  
   Chongjiong Fan, Zhicheng Pan, Wenwen Sun, et al. *ECML PKDD 2024.* [[Paper](https://doi.org/10.1007/978-3-031-70362-1_22)]
4. **Is Large Language Model Good at Database Knob Tuning? A Comprehensive Experimental Evaluation**  
   Yiyan Li, Haoyang Li, Zhao Pu, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2408.02213)]

##### RAG Based Tuning Experience Enrichment

1. **Automatic Database Configuration Debugging using Retrieval-Augmented Language Models**  
   Sibei Chen, Ju Fan, Bin Wu, et al. *Proceedings of the ACM on Management of Data, Volume 3, Issue 1, 2025.* [[Paper](https://dl.acm.org/doi/10.1145/3709663)]
2. **GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization**  
   Jiale Lao, Yibo Wang, Yufei Li, et al. *VLDB 2024.* [[Paper](https://doi.org/10.14778/3659437.3659449)]

##### Training Enhanced Tuning Goal Alignment

1. **E2ETune: End-to-End Knob Tuning via Fine-tuned Generative Language Model**  
   Xinmei Huang, Haoyang Li, Jing Zhang, et al. *VLDB 2025.* [[Paper](https://doi.org/10.48550/arXiv.2404.11581)]
2. **DB-GPT: Large Language Model Meets Database**  
   Xuanhe Zhou, Zhaoyan Sun, Guoliang Li. *Data Science and Engineering 2024.* [[Paper](https://link.springer.com/article/10.1007/s41019-023-00235-6)]
3. **HEBO: Heteroscedastic Evolutionary Bayesian Optimisation**  
   Alexander I. Cowen-Rivers, Wenlong Lyu, Zhi Wang, et al. *NeurIPS 2020*. [[Paper](https://arxiv.org/abs/2012.03826v1)]



#### 4.2.2 LLM for Query Optimization

##### Optimization-Aware Prompt Engineering

1. **E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency**    
   Dongjie Xu, Yue Cui, Weijie Shi, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2508.09023v2)]
2. **LLM4Hint: Leveraging Large Language Models for Hint Recommendation in Offline Query Optimization**    
   Suchen Liu, Jun Gao, Yinjun Han, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2507.03384v1)]
3. **QUITE: A Query Rewrite System Beyond Rules with LLM Agents**  
   Yuyang Song, Hanxu Yan, Jiale Lao, et al. *arXiv 2025.* [[Paper](https://arxiv.org/pdf/2506.07675)]
4. **Can Large Language Models Be Query Optimizer for Relational Databases?**  
   Jie Tan, Kangfei Zhao, Rui Li, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.05562)]
5. **A Query Optimization Method Utilizing Large Language Models**  
   Zhiming Yao, Haoyang Li, Jing Zhang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2503.06902)]
6. **Query Rewriting via LLMs**  
   Sriram Dharwada, Himanshu Devrani, Jayant Haritsa, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.12918)]
7. **DB-GPT: Large Language Model Meets Database** [[Paper](https://link.springer.com/article/10.1007/s41019-023-00235-6)]
8. **LLM-R2: A Large Language Model Enhanced Rule-Based Rewrite System for Boosting Query Efficiency**  
   Zhaodonghui Li, Haitao Yuan, Huiming Wang, et al. *VLDB 2024.* [[Paper](https://doi.org/10.14778/3696435.3696440)]
9. **The Unreasonable Effectiveness of LLMs for Query Optimization**  
     Peter Akioyamen, Zixuan Yi, Ryan Marcus. *ML for Systems Workshop at NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2411.02862)]
10. **R-Bot: An LLM-based Query Rewrite System**  
   Zhaoyan Sun, Xuanhe Zhou, Guoliang Li. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.01661)]
11. **Query Rewriting via Large Language Models**  
      Jie Liu, Barzan Mozafari. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.09060)]



#### 4.2.3 LLM for Anomaly Diagnosis

##### Manually Crafted Prompts for Anomaly Diagnosis

1. **DBG-PT: A Large Language Model Assisted Query Performance Regression Debugger**  
   Victor Giannakouris, Immanuel Trummer. *Proceedings of the VLDB Endowment, Volume 17, Issue 12, 2024.* [[Paper](https://doi.org/10.14778/3685800.3685869)]

##### RAG Based Diagnosis Experience Enrichment

1. **DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs**     
   Wei Zhou, Peng Sun, Xuanhe Zhou, et al. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2508.01136)]
2. **Query Performance Explanation through Large Language Model for HTAP Systems**   
   Haibo Xiu, Li Zhang, Tieying Zhang, et al. *ICDE 2025.* [[Paper](https://doi.org/10.48550/arXiv.2412.01709)]
3. **D-Bot: Database Diagnosis System using Large Language Models**  
   Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
4. **LLM As DBA**  
   Xuanhe Zhou, Guoliang Li, Zhiyuan Liu. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2308.05481)]

##### Multi-Agent Mechanism for Collaborative Diagnosis

1. **GaussMaster: An LLM-based Database Copilot System**  
   Wei Zhou, Ji Sun, Xuanhe Zhou, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2506.23322)]
2. **D-Bot: Database Diagnosis System using Large Language Models** [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
3. **Panda: Performance Debugging for Databases using LLM Agents**  
   Vikramank Singh, Kapil Eknath Vaidya, Vinayshekhar Bannihatti Kumar, et al. *CIDR 2024.* [[Paper](https://www.cidrdb.org/cidr2024/papers/p6-singh.pdf)]
7. **LLM As DBA** [[Paper](https://arxiv.org/abs/2308.05481)]

##### Localized LLM Enhancement via Specialized FineTuning

1. **D-Bot: Database Diagnosis System using Large Language Models** [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
3. **LLM for Data Management**     
   Guoliang Li, Xuanhe Zhou, Xinyang Zhao. *PVLDB 17(12).* 2024. [[Paper](https://doi.org/10.14778/3685800.3685838)]
4. **LLM-Enhanced Data Management**     
   Xuanhe Zhou, Xinyang Zhao, Guoliang Li. *arXiv 2024*. [[Paper](https://doi.org/10.48550/arXiv.2402.02643)]

## 5 LLM as Data Analyst

### 5.1 LLM for Structured Data Analysis

#### 5.1.1 Relational Data

1. **A relational model of data for large shared data banks.** [[Paper](https://dl.acm.org/doi/abs/10.1145/362384.262685)]

2. **Multilinear tensor regression for longitudinal relational data** [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4957660/)]

3. **Probabilistic classification and clustering in relational data** [[Paper](http://robotics.stanford.edu/~koller/Papers/Taskar+al:IJCAI01.pdf)]

4. **Outlier detection in relational data: A case study in geographical information systems** [[Paper](https://www.sciencedirect.com/science/article/pii/S0957417411014485)]

##### NL2SQL

1. **Finsql: Model-agnostic llms-based text-to-sql framework for financial analysis** [[Paper](https://dl.acm.org/doi/abs/10.1145/3626246.3653375)]

2. **Pet-sql: A prompt-enhanced two-round refinement of text-to-sql with cross-consistency** [[Paper](https://arxiv.org/abs/2403.09732)]

3. **Chess: Contextual harnessing for efficient sql synthesis** [[Paper](https://arxiv.org/abs/2405.16755)]

4. **Codes: Towards building open-source language models for text-to-sql** [[Paper](https://dl.acm.org/doi/abs/10.1145/3654930)]

5. **Combining small language models and large language models for zero-shot nl2sql** [[Paper](https://dl.acm.org/doi/abs/10.14778/3681954.2681960)]

6. **Cracking SQL Barriers: An llm-based dialect translation system** [[Paper](https://dl.acm.org/doi/abs/10.1145/3725278)]

7. **Cracksql: A hybrid sql dialect translation system powered by large language models** [[Paper](https://arxiv.org/abs/2504.00882)]

8. **Din-sql: Decomposed in-context learning of text-to-sql with self-correction** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html)]

9. **Opensearch-sql: Enhancing text-to-sql with dynamic few-shot and consistency alignment** [[Paper](https://dl.acm.org/doi/abs/10.1145/3725331)]

10. **Bridging the semantic gap between text and table: A case study on nl2sql** [[Paper](https://openreview.net/forum?id=qmsX2R19p9)]

11. **The dawn of natural language to sql: Are we fully ready?** [[Paper](https://arxiv.org/abs/2406.01265)]

12. **A Survey of Text-to-SQL in the Era of LLMs: Where Are We, and Where Are We Going?** [[Paper](https://ieeexplore.ieee.org/abstract/document/11095853/)]

13. **Natural Language to SQL: State of the Art and Open Problems** [[Paper]([https://ieeexplore.ieee.org/abstract/document/11095853/](https://dl.acm.org/doi/abs/10.14778/3750601.3750696))]

14. **A survey on employing large language models for text-to-sql tasks** [[Paper]([https://ieeexplore.ieee.org/abstract/document/11095853/](https://dl.acm.org/doi/abs/10.1145/3737873))]

##### NL2Code

1. **Natural language to code generation in interactive data science notebooks** [[Paper](https://arxiv.org/abs/2212.09248)]

2. Palm: Scaling language modeling with pathways [[Paper](https://www.jmlr.org/papers/v24/22-1144.html)]

3. Contextualized data-wrangling code generation in computational notebooks [[Paper](https://dl.acm.org/doi/abs/10.1145/3691620.3695503)]

4. Data interpreter: An llm agent for data science [[Paper](https://arxiv.org/abs/2402.18679)]

5. Collaboration between intelligent agents and large language models: A novel approach for enhancing code generation capability [[Paper](https://www.sciencedirect.com/science/article/pii/S095741742403224X)]

6. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. [[Paper](https://arxiv.org/abs/1910.13461?spm=a2c6h.13046898.publish-article.9.99fc6ffa8EPYJl&file=1910.13461)]

##### LLM for Semantic Analysis.

**Multi-Step QA.**

1. Tat-llm: A specialized language model for discrete reasoning over financial tab- ular and textual data [[Paper](https://dl.acm.org/doi/abs/10.1145/3677052.3698685)]

2. S3HQA: A three-stage approach for multi-hop text-table hybrid question answering [[Paper](https://arxiv.org/abs/2305.11725)]

3. Plugging schema graph into multi-table qa: A human-guided framework for reducing llm reliance. [[Paper](https://arxiv.org/abs/2506.04427)]

4. TaPERA: Enhancing faithfulness and interpretability in long-form table QA by content planning and execution-based reasoni [[Paper](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2024.acl-long.692.pdf)]

5. Reactable: Enhancing react for table question answering [[Paper](https://dl.acm.org/doi/abs/10.14778/3659437.3659452)]

6. Chain-of-table: Evolving tables in the reasoning chain for table understanding [[Paper](https://arxiv.org/abs/2401.04398)]

**End-to-End QA**

1. Table-gpt: Table-tuned gpt for diverse table tasks [[Paper](https://dl.acm.org/doi/abs/10.1145/3654979)]

2. Tablegpt2: A large multimodal model with tabular data integration [[Paper](https://arxiv.org/abs/2411.02059)]

3. Cabinet: Content relevance based noise reduction for table question answering [[Paper](https://arxiv.org/abs/2402.01155)]

4. Tablemaster: A recipe to advance table understanding with language models [[Paper](https://arxiv.org/abs/2501.19378)]

5. Mmqa: Evaluating llms with multi-table multi-hop complex questions. [[Paper](https://openreview.net/pdf?id=GGlpykXDCa)]

6. Multimodal table understanding [[Paper](https://arxiv.org/abs/2406.08100)]

7. Improved baselines with visual instruction tuning [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html)]

8. Tabpedia: Towards comprehensive visual table understanding with concept synergy [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0d97fe65d7a1dc12a05642d9fa4cd578-Abstract-Conference.html)]

9. Judging llm-as-a-judge with mt-bench and chatbot arena. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

##### LLM for Time Series Analysis.

1. Time series databases and influxdb [[Paper](https://www.devopsschool.com/blog/wp-content/uploads/2022/09/influxdb_2017.pdf)]

2. Towards cross-modality modeling for time series analytics: A survey in the llm era [[Paper](https://arxiv.org/abs/2505.02583)]

3. A comparison of arima and lstm in forecasting time series [[Paper](https://ieeexplore.ieee.org/abstract/document/8614252)]

4. Association between forecasting models’ precision and nonlinear patterns of daily river flow time series [[Paper](https://link.springer.com/article/10.1007/s40808-022-01351-4)]

5. The performance of lstm and bilstm in forecasting time series [[Paper](https://ieeexplore.ieee.org/abstract/document/9005997)]

6. Hmckrautoencoder: An interpretable deep learning framework for time series analysis. [[Paper](https://ieeexplore.ieee.org/abstract/document/9713986)]

**TS2NL.**

1. Can large language models be anomaly detectors for time series? [[Paper](https://ieeexplore.ieee.org/abstract/document/10722786)]

2. Timerag: Boosting llm time series forecasting via retrieval-augmented generation.
[[Paper](https://ieeexplore.ieee.org/abstract/document/10889933/)]

3. Dynamic time warping algorithm review. [[Paper](https://www.researchgate.net/profile/Pavel-Senin/publication/228785661_Dynamic_Time_Warping_Algorithm_Review/links/02bfe5100f11a7929f000000/Dynamic-Time-Warping-Algorithm-Review.pdf)]

4. Temporal data meets llm–explainable financial time series forecasting. [[Paper](https://arxiv.org/abs/2306.11025)]

5. Exploring large language models for climate forecasting [[Paper](https://arxiv.org/abs/2411.13724)]

6. Timecap: Learning to contextualize, augment, and predict time series events with large language model agents [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33989)]

7. Explainable multi-modal time series prediction with llm-in-the-loop [[Paper](https://arxiv.org/abs/2503.01013)]

8. From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6aef8bffb372096ee73d98da30119f89-Abstract-Conference.html)]

**Alignment.**

1. Time-llm: Time series forecasting by reprogramming large language models [[Paper](https://arxiv.org/abs/2310.01728)]

2. Seed: A structural encoder for embedding-driven decoding in time series prediction with llms [[Paper](https://arxiv.org/abs/2506.20167)]

3. Timecma: Towards llm-empowered multivariate time series forecasting via cross-modality alignment [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34067)]

4. Calf: Aligning llms for time series forecasting via cross-modal fine-tuning [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34082)]

5. S2IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting [[Paper](https://openreview.net/forum?id=qwQVV5R8Y7)]

6. Llm4ts: Aligning pre-trained llms as data-efficient time-series forecasters [[Paper](https://dl.acm.org/doi/abs/10.1145/3719207)]

7. Large language models are few-shot multivariate time series classifiers. [[Paper](https://link.springer.com/article/10.1007/s10618-025-01145-z)]

#### 5.1.2 Graph Data Analysis
1. A comparison of current graph database models [[Paper](https://ieeexplore.ieee.org/abstract/document/6313676)]

**Natural Language To Graph Analysis Query.**

1. Nat-nl2gql: A novel multi-agent framework for translating natural language to graph query language [[Paper](https://arxiv.org/abs/2412.10434)]

2. r3-NL2GQL: A model coordination and knowledge graph alignment approach for NL2GQL [[Paper](https://arxiv.org/abs/2311.01862)]

3. Aligning large language models to a domain-specific graph database for nl2gql [[Paper](https://dl.acm.org/doi/abs/10.1145/3627673.3679713)]

4. Graph learning in the era of llms: A survey from the perspective of data, models, and tasks [[Paper](https://arxiv.org/abs/2412.12456)]

5. Leveraging biomolecule and natural language through multi-modal learning: A survey [[Paper](https://arxiv.org/abs/2403.01528)]

**LLM-based Semantic Analysis.**

- Retrieval-Then-Reasoning.

1. Subgraph retrieval enhanced model for multi-hop knowledge base question answering [[Paper](https://arxiv.org/abs/2202.13296)]

2. Unikgqa: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph [[Paper](https://arxiv.org/abs/2212.00959)]

3. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)]

- Execution-Then-Reasoning

1. Interactive-kbqa: Multi-turn inter-actions for knowledge base question answering with large language models [[Paper](https://arxiv.org/abs/2402.15131)]

2. Mcts-kbqa: Monte carlo tree search for knowledge base question answering [[Paper](https://arxiv.org/abs/2502.13428)]

3. Flexkbqa: A flexible llm-powered framework for few-shot knowledge base question answering [[Paper]()]https://ojs.aaai.org/index.php/AAAI/article/view/29823

**Graph Task Based Fine-tuning Methods.**

1. Language is all a graph needs [[Paper](https://arxiv.org/abs/2308.07134)]

2. Instruct-graph: Boosting large language models via graph-centric instruction tuning and preference alignment [[Paper](https://arxiv.org/abs/2402.08785)]

3. Direct preference optimization: Your language model is secretly a reward model [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)]

4. Graphgpt: Graph instruction tuning for large language models [[Paper](https://dl.acm.org/doi/abs/10.1145/3626772.3657775)]

5. Inductive representation learning on large graphs [[Paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)]

6. Semi-supervised classification with graph convolutional networks. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.html)]

7. Glam: Fine-tuning large language models for domain knowledge graph alignment via neighborhood partitioning and generative sub-graph encoding [[Paper](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31186
)]

- Agent Based Methods.

1. Structgpt: A general framework for large language model to reason over structured data [[Paper](https://arxiv.org/abs/2305.09645)]

2. Kbqa-o1: Agentic knowledge base question answering with monte carlo tree search. [[Paper](https://arxiv.org/abs/2501.18922)]

3. Call me when necessary: Llms can efficiently and faithfully reason over structured environments [[Paper](https://arxiv.org/abs/2403.08593)]

#### 5.1.3 Structured Data Generation for LLM

1. Compositional Semantic Parsing on Semi-Structured Tables [[Paper](https://arxiv.org/abs/1508.00305)]

2. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task [[Paper](https://arxiv.org/abs/1809.08887)]

##### Relational Data Generation.

1. REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers [[Paper](https://arxiv.org/abs/2302.02041)]

2. Relational data generation with graph neural networks and latent diffusion models [[Paper](https://openreview.net/forum?id=MNLR2NYN2Z)]

3. Synthetic data generation of many-to-many datasets via random graph generation. [[Paper](https://openreview.net/forum?id=Q120_4COf-K)]

4. Mixed-type tabular data synthesis with score-based diffusion in latent space [[Paper](https://arxiv.org/abs/2310.09656)]

5. Syntaxsqlnet: Syntax tree networks for complex and cross-domain text-to-sql task [[Paper](https://arxiv.org/abs/1810.05237)]

6. Codes: Towards building open-source language models for text-to-sql [[Paper](https://dl.acm.org/doi/abs/10.1145/3654930)]

7. Itf-gan: Synthetic time series dataset generation and manipulation by interpretable features [[Paper](https://www.sciencedirect.com/science/article/pii/S095070512300881X)]

8. ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning [[Paper](https://arxiv.org/abs/2412.03104)]

##### Graph Data Generation. 

1. A framework for large-scale synthetic graph dataset generation [[Paper](https://ieeexplore.ieee.org/abstract/document/10943225)]

2. A temporal knowledge graph generation dataset supervised distantly by large language models [[Paper](https://www.nature.com/articles/s41597-025-05062-0)]



### 5.2 LLM for Semi-Structured Data Analysis

#### 5.2.1 Markup Language

**Markup Extraction.**

1. Language models enable simple systems for generating structured views of heterogeneous data lakes
[[Paper](https://arxiv.org/abs/2304.09433)]

2. Webformer: The web-page transformer for structure information extraction
[[Paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512032)]

**Markup Query.**

1. XPath Agent: An Efficient XPath Programming Agent Based on LLM for Web Crawler
[[Paper](https://arxiv.org/abs/2502.15688)]

2. Bridging the gap: Enabling natural language queries for nosql databases through text-to-nosql translation
[[Paper](https://arxiv.org/abs/2502.11201)]

**Markup Understanding.**

1. Dom-lm: Learning generalizable representations for html documents
[[Paper](https://arxiv.org/abs/2201.10608)]

2. Markuplm: Pre-training of text and markup language for visually-rich document understanding
[[Paper](https://arxiv.org/abs/2110.08518)]

3. Hierarchical multimodal pre-training for visually rich webpage understanding
[[Paper](https://dl.acm.org/doi/abs/10.1145/3616855.3635753)]

#### 5.2.2 Semi-Structured Table

**Table Representation.**

1. Tuta: Tree-based transformers for generally structured table pre-training
[[Paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467434)]

2. ST-Raptor: LLM-Powered Semi-Structured Table Question Answering
[[Paper](https://arxiv.org/abs/2508.18190)]

3. Reasoning and Retrieval for Complex Semi-structured Tables via Reinforced Relational Data Transformation
[[Paper](https://dl.acm.org/doi/abs/10.1145/3726302.3730071)]

4. Auto-Tables: Synthesizing Multi-Step Transformations to Relationalize Tables without Using Examples
[[Paper](https://arxiv.org/abs/2307.14565)]

5. Can an LLM find its way around a Spreadsheet?
[[Paper](https://vtechworks.lib.vt.edu/items/51d233b7-b566-4e55-9ba6-8f5d7ca8a256)]

**Table Prompting.**

1. SpreadsheetLLM: encoding spreadsheets for large language models
[[Paper](https://arxiv.org/abs/2407.09025)]

2. HySem: A context length optimized LLM pipeline for unstructured tabular extraction
[[Paper](https://openreview.net/forum?id=1f6SIVYExy#discussion)]

**Table Querying.**

1. SpreadsheetLLM: encoding spreadsheets for large language models
[[Paper](https://arxiv.org/abs/2407.09025)]

2. ST-Raptor: LLM-Powered Semi-Structured Table Question Answering
[[Paper](https://arxiv.org/abs/2508.18190)]

### 5.3 LLM for Unstructured Data Analysis

#### 5.3.1 Chart Analysis

**Traditional Approaches**
1. DVQA: Understanding Data Visualizations viaQuestion Answering
[[Paper](https://arxiv.org/abs/1801.08163)]

**Chart Captioning**
1. Describing Complex Charts in Natural Language A Caption Generation System
[[Paper](https://aclanthology.org/J98-3004/)]

2. An Architecture for Data-to-Text Systems[[Paper](https://aclanthology.org/W07-2315/)]

3. Chartthinker: A contextual chain-of-thought approach to optimized chart summarization[[Paper](https://arxiv.org/abs/2403.11236)]

4. Chart-to-Text: Generating Natural Language Descriptions for Charts by Adapting the Transformer Model[[Paper](https://aclanthology.org/2020.inlg-1.20/)]

5. FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback[[Paper](https://arxiv.org/abs/2307.10867)]

6. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning[[Paper](https://arxiv.org/abs/2305.14761)]

**Chart Question Answering**
1. ChartLlama: A Multimodal LLM for Chart Undestanding and Generation
[[Paper](https://arxiv.org/abs/2311.16483)]

2. ChartBench: A Benchmark for Complex Visual easoning in Charts
[[Paper](https://arxiv.org/abs/2312.15915)]

3. Evochart: A benchmark and a self-training approach towards real-world chart understanding[[Paper](https://arxiv.org/abs/2409.01577)]

4. Chartinsights: Evaluating multimodal large language models for low-level chart question answering[[Paper](https://arxiv.org/abs/2405.07001)]

5. Vizability: Enhancing chart accessibility with llm-based conversational interaction[[Paper](https://arxiv.org/abs/2310.09611)]

6. Charts-of-Thought: Enhancing LLM Visualization Literacy Through Structured Data Extraction[[Paper](https://arxiv.org/abs/2508.04842)]

7. ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding[[Paper](https://arxiv.org/abs/2409.03277)]

8. ChartGemma: Visual Instruction-tuning for Cart Reasoning in the Wild
[[Paper](https://arxiv.org/abs/2407.04172)]

9. mPLUG-Owl: Modularization Empowers Large Laguage Models with Multimodality
[[Paper](https://arxiv.org/abs/2304.14178)]

**Chart-to-Code**
1. ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation[[Paper](https://arxiv.org/abs/2406.09961)]

2. Text2Chart31: Instruction Tuning for Chart Generation with Automatic Feedback[[Paper](https://arxiv.org/abs/2410.04064)]

3. Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation[[Paper](https://arxiv.org/abs/2508.13587)]

#### 5.3.2 Video Analysis
##### Temporally-Anchored Approaches

1. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability [[Paper](https://arxiv.org/abs/2411.18211)]

2. Seq2time: Sequential knowledge transfer for video llm temporal grounding [[Paper](https://arxiv.org/abs/2411.16932)]

3. Tempme: Video temporal token merging for efficient text-video retrieval [[Paper](https://arxiv.org/abs/2409.01156)]

4. Video token merging for long-form video understanding [[Paper](https://arxiv.org/abs/2410.23782)]

5. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models [[Paper](https://arxiv.org/abs/2410.03290)]

##### Instruction-Aware Relative Temporal Localization

1. From image to video, what do we need in multimodal llms? [[Paper](https://arxiv.org/abs/2404.11865)]

2. LLMs meet long video: Advancing long video comprehension with an interactive visual adapter in llms [[Paper](https://arxiv.org/abs/2402.13546)]


##### Video Emotional Analysis

1. Predicting Team Well-Being through Face Video Analysis with AI [[Paper](https://www.mdpi.com/2076-3417/14/3/1284)]

2. AI based multimodal emotion and behavior analysis of interviewee [[Paper](https://www.researchgate.net/profile/Vijaya-Bharathi-Jagan/publication/370653388_I_NTERNATIONAL_J_OURNAL_OF_S_CIENTIFIC_R_ESEARCH_IN_E_NGINEERING_AND_M_ANAGEMENT_IJSREM_AI_Based_Multimodal_Emotion_and_Behavior_Analysis_of_Interviewee/links/645bd6d2f43b8a29ba40d2f8/I-NTERNATIONAL-J-OURNAL-OF-S-CIENTIFIC-R-ESEARCH-IN-E-NGINEERING-AND-M-ANAGEMENT-IJSREM-AI-Based-Multimodal-Emotion-and-Behavior-Analysis-of-Interviewee.pdf)]

##### Object Detection

1. Videorefer suite: Advancing spatial-temporal object understanding with video llm [[Paper](https://arxiv.org/abs/2501.00599)]

2. Video summarisation with incident and context information using generative ai [[Paper](https://arxiv.org/abs/2501.04764)]

3. Abnormal event detection in surveillance videos through LSTM auto-encoding and local minima assistance [[Paper](https://link.springer.com/content/pdf/10.1007/s43926-025-00127-3.pdf)]

##### Gesture and Behavior Detection

1. Utilizing multimodal large language models for video analysis of posture in studying collaborative learning: A case study [[Paper](https://learning-analytics.info/index.php/JLA/article/view/8595)]

2. Artificial intelligence–powered 3D analysis of video-based caregiver-child interactions [[Paper](https://www.science.org/doi/full/10.1126/sciadv.adp4422)]

##### Video Data for LLM

1. VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding [[Paper](https://arxiv.org/abs/2507.13353)]

2. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis [[Paper](https://arxiv.org/abs/2207.09814)]

3. Text2video-zero: Text-to-image diffusion models are zero-shot video generators [[Paper](https://arxiv.org/abs/2303.13439)]

4. Align your latents: High-resolution video synthesis with latent diffusion models [[Paper](https://arxiv.org/abs/2304.08818)]

5. Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation [[Paper](https://arxiv.org/abs/2211.12194)]

6. DreamTalk: When Emotional Talking Head Generation Meets Diffusion Probabilistic Models [[Paper](https://arxiv.org/abs/2312.09767)]

7. Disco: Disentangled control for realistic human dance generation [[Paper](https://arxiv.org/abs/2307.00040)]

8. Imagen video: High definition video generation with diffusion models [[Paper](https://arxiv.org/abs/2210.02303)]

9. Make-a-video: Text-to-video generation without text-video data [[Paper](https://arxiv.org/abs/2209.14792)]


#### 5.3.3 Document Analysis

1. M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding. Jaemin Cho et al.. arXiv preprint arXiv:2411.04952 2024. [https://arxiv.org/abs/2411.04952]
2. SV-RAG: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding. Jian Chen et al.. arXiv preprint arXiv:2411.01106 2024. [https://arxiv.org/abs/2411.01106]
3. VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding. Ofir Abramovich et al.. arXiv preprint arXiv:2407.12594 2024. [https://arxiv.org/abs/2407.12594]
4. DocFormer: End-to-End Transformer for Document Understanding. Appalaraju, Srikar et al.. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2021. [https://arxiv.org/abs/2106.11539]
5. VLCDoC: Vision-Language contrastive pre-training model for cross-Modal document classification. Sahar Bakkali et al.. Pattern Recognition 2023. [https://arxiv.org/abs/2205.12029]
6. Efficient End-to-End Visual Document Understanding with Rationale Distillation. Wang Zhu et al.. arXiv preprint arXiv:2412.08519 2024. [https://arxiv.org/abs/2412.08519]
7. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. Yupan Huang et al.. Proceedings of the 30th ACM International Conference on Multimedia 2022. [https://arxiv.org/abs/2204.08387]
8. MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection. Ting-Yu Huang, Shan-Yu Chuang, Li-Heng Sun, Min-Hung Chen. arXiv preprint arXiv:2410.03010 2024. [https://arxiv.org/abs/2410.03010]
9. Unifying Layout Generation with a Decoupled Diffusion Model. Mude Hui et al.. arXiv preprint arXiv:2303.05049 2023. [https://arxiv.org/abs/2303.05049]
10. LayoutDM: Discrete Diffusion Model for Controllable Layout Generation. Inoue, Naoto et al.. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023. [https://arxiv.org/abs/2303.08137]
11. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran. 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW) 2019. [https://arxiv.org/abs/1905.13538]
12. ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision. Wonjae Kim, Bokyung Son, Ildoo Kim. International Conference on Machine Learning 2021. [https://arxiv.org/abs/2102.03334]
13. VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation. Manan Suri et al.. arXiv preprint arXiv:2407.12733 2024. [https://arxiv.org/abs/2407.12733]
14. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Patrick Lewis et al.. Advances in Neural Information Processing Systems 2020. [https://arxiv.org/abs/2005.11401]
15. MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis. Mengmeng Ma et al.. Transactions of the Association for Computational Linguistics 2023. [DOI:10.1162/tacl_a_00628](https://doi.org/10.1162/tacl_a_00628)
16. CREPE: Coordinate-Aware End-to-End Document Parser. Yamato Okamoto et al.. arXiv preprint arXiv:2405.00260 2024. [https://arxiv.org/abs/2405.00260]
17. LTSim: Layout Transportation-based Similarity Measure for Evaluating Layout Generation. Mayu Otani, Naoto Inoue, Kotaro Kikuchi, Riku Togashi. arXiv preprint arXiv:2407.12356 2024. [https://arxiv.org/abs/2407.12356]
18. AesthetiQ: Enhancing Graphic Layout Design via Aesthetic-Aware Preference Alignment of Multi-modal Large Language Models. Sohan Patnaik, Rishabh Jain, Balaji Krishnamurthy, Mausoom Sarkar. arXiv preprint arXiv:2503.00591 2025. [https://arxiv.org/abs/2503.00591]
19. Automatic generation of scientific papers for data augmentation in document layout analysis. Marco Pisaneschi, Srikar Appalaraju, R. Manmatha. Pattern Recognition Letters 2023. [DOI:10.1016/j.patrec.2023.01.011](https://doi.org/10.1016/j.patrec.2023.01.011)
20. PosterLlama: Bridging Design Ability of Language Model to Content-Aware Layout Generation. Jaejung Seol, Seojun Kim, Jaejun Yoo. European Conference on Computer Vision (ECCV) 2024. [https://arxiv.org/abs/2404.00995]
21. LayoutCoT: Chain-of-Thought Prompting for Layout Generation. Shang Chai, Liansheng Zhuang, Fengying Yan. arXiv preprint arXiv:2504.10829 2024. [https://arxiv.org/abs/2504.10829]
22. SciPostLayout: A Dataset for Layout Analysis and Layout Generation of Scientific Posters. Shohei Tanaka, Hao Wang, Yoshitaka Ushiku. arXiv preprint arXiv:2407.19787 2024. [https://arxiv.org/abs/2407.19787]
23. OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition. Jianqiang Wan et al.. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2024. [https://arxiv.org/abs/2403.19128]
24. DLAFormer: An End-to-End Transformer For Document Layout Analysis. Jiawei Wang, Kai Hu, Qiang Huo. arXiv preprint arXiv:2405.11757 2024. [https://arxiv.org/abs/2405.11757]
25. DocLLM: A layout-aware generative language model for multimodal document understanding. Dongsheng Wang et al.. arXiv preprint arXiv:2401.00908 2024. [https://arxiv.org/abs/2401.00908]
26. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. Yiheng Xu et al.. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining 2020. [https://arxiv.org/abs/1912.13318]
27. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. Yang Xu et al.. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) 2021. [https://arxiv.org/abs/2012.14740]
28. Corrective Retrieval Augmented Generation. Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling. arXiv preprint arXiv:2401.15884 2024. [https://arxiv.org/abs/2401.15884]
29. RAFT: Adapting Language Model to Domain Specific RAG. Tianjun Zhang et al.. arXiv preprint arXiv:2403.10131 2024. [https://arxiv.org/abs/2403.10131]
30. VASCAR: Content-Aware Layout Generation via Visual-Spatial Self-Correction. Chengyou Jia et al.. arXiv preprint arXiv:2412.04237 2024. [https://arxiv.org/abs/2412.04237]

#### 5.3.4 Program Analysis

1. Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization). Ahmed, Toufique, Pai, Kunal Suresh, Devanbu, Premkumar, Barr, Earl. Proceedings of the IEEE/ACM 46th International Conference on Software Engineering 2024. [DOI:10.1145/3597503.3639183](https://doi.org/10.1145/3597503.3639183)
2. Teaching Large Language Models to Self-Debug. Chen, Xinyun, Lin, Maxwell I., Schärli, Nathanael, Zhou, Denny. The Eleventh International Conference on Learning Representations (ICLR 2023) 2023. [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)
3. Syntax-directed variational autoencoder for structured data. Dai, Hanjun et al.. International Conference on Learning Representations 2018. [https://arxiv.org/abs/1802.08786]
4. Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning. Mingyang Geng et al.. misc 2023. [arXiv:2304.11384](https://arxiv.org/abs/2304.11384)
5. FT2Ra. Guo, Qi et al.. Proceedings of the 46th IEEE/ACM International Conference on Software Engineering (ICSE '24) 2024. [arXiv:2402.00862](https://arxiv.org/abs/2402.00862)
6. Composing graphical models with neural networks for structured representations and fast inference. Johnson, Matthew J et al.. Advances in neural information processing systems 2016. [https://arxiv.org/abs/1603.06277]
7. Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning. {Numina, Kimi Team. arXiv preprint arXiv:2504.11354 2025. [https://arxiv.org/abs/2504.11354]
8. REPOFUSE. Liang, Ming et al.. arXiv preprint arXiv:2402.14275 2024. [arXiv:2402.14275](https://arxiv.org/abs/2402.14275)
9. G\"o. Lin, Yong et al.. arXiv preprint arXiv:2502.07640 2025. [https://arxiv.org/abs/2502.07640]
10. Software Vulnerability Detection with GPT and In-Context Learning. Liu, Zhihong, Liao, Qing, Gu, Wenchao, Gao, Cuiyun. 2023 8th International Conference on Data Science in Cyberspace (DSC) 2023. [DOI:10.1109/DSC59305.2023.00041](https://doi.org/10.1109/DSC59305.2023.00041)
11. Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks. Zhongxin Liu et al.. misc 2024. [arXiv:2402.00657](https://arxiv.org/abs/2402.00657)
12. WizardCoder: Empowering Code Large Language Models with Evol-Instruct. Ziyang Luo et al.. misc 2023. [arXiv:2306.08568](https://arxiv.org/abs/2306.08568)
13. SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation. Yingjie Mao et al.. misc 2024. [arXiv:2402.04863](https://arxiv.org/abs/2402.04863)
14. Self-Instruct: Aligning Language Models with Self-Generated Instructions. Yizhong Wang et al.. misc 2022. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
15. Magicoder: Empowering Code Generation with OSS-Instruct. Yuxiang Wei et al.. misc 2023. [arXiv:2312.02120](https://arxiv.org/abs/2312.02120)
16. Repoformer. Wu, Di, Ahmad, Wasi Uddin, Li, Shanshan, Ma, Xiaofei. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '24) 2024. [arXiv:2403.09858](https://arxiv.org/abs/2403.09858)
17. DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data. Xin, Huajian et al.. arXiv preprint arXiv:2405.14333 2024. [https://arxiv.org/abs/2405.14333]
18. Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code. Zhang, Junwei et al.. IEEE Transactions on Software Engineering 2023. [DOI:10.1109/TSE.2023.3286586](https://doi.org/10.1109/TSE.2023.3286586)
19. RepoCoder. Zhang, Fengji et al.. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing 2023. [arXiv:2303.12570](https://arxiv.org/abs/2303.12570)
20. Large Language Model for Vulnerability Detection: Emerging Results and Future Directions. Xin Zhou, Ting Zhang, David Lo. misc 2024. [arXiv:2401.15468](https://arxiv.org/abs/2401.15468)


#### 5.3.5 3D Model Analysis

##### 3D-Language Fusion
1. 3d-llm: Injecting the 3d world into large language models [[Paper](https://arxiv.org/abs/2307.12981)]

2. 3ur-llm: An end-to-end multimodal large language model for 3d scene understanding [[Paper](https://arxiv.org/abs/2501.07819)]

3. Towards 3d molecule-text interpretation in language models [[Paper](https://arxiv.org/abs/2401.13923)]

4. Proteinchat: Towards achieving chatgpt-like functionalities on protein 3d structures [[Paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23120606)]

5. Protchatgpt: Towards understanding proteins with large language models [[Paper](https://arxiv.org/abs/2402.09649)]

##### 3D-Derived Task Enhancement
1. Do Large Language Models Truly Understand Geometric Structures? [[Paper](https://arxiv.org/abs/2501.13773)]

2. 3DSMILES-GPT: 3D molecular pocket-based generation with token-only large language model [[Paper](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc06864e)]

3. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules [[Paper](https://pubs.acs.org/doi/pdf/10.1021/ci00057a005)]

4. ProtChat: An AI Multi-Agent for Automated Protein Analysis Leveraging GPT-4 and Protein Language Model [[Paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01345)]

5. A multimodal protein representation framework for quantifying transferability across biochemical downstream tasks [[Paper](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202301223)]

##### Cross-modal Capability Refinement
1. Self-supervised image-based 3d model retrieval [[Paper](https://dl.acm.org/doi/abs/10.1145/3548690)]

2. Llmi3d: Empowering llm with 3d perception from a single 2d image [[Paper](https://arxiv.org/abs/2408.07422)]

##### 3-D data for LLM
1. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation [[Paper](https://arxiv.org/abs/2303.13873)]

2. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d [[Paper](https://arxiv.org/abs/2310.02596)]

3. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d [[Paper](https://arxiv.org/abs/2311.16918)]

4. Zero-1-to-3: Zero-shot one image to 3d object [[Paper](https://arxiv.org/abs/2303.11328)]

5. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation [[Paper](https://arxiv.org/abs/2411.02293)]

6. Craftsman3d: High-fidelity mesh generation with 3d native generation and interactive geometry refiner [[Paper](https://arxiv.org/abs/2405.14979)]

7. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer [[Paper](https://arxiv.org/abs/2405.14832)]

8. Meshanything: Artist-created mesh generation with autoregressive transformers [[Paper](https://arxiv.org/abs/2406.10163)]

9. Llama-mesh: Unifying 3d mesh generation with language models [[Paper](https://arxiv.org/abs/2411.09595)]


### 5.4 LLM for Heterogeneous Data Analysis

#### 5.4.1 LLM for Modality Alignment

1. Unicorn: a unified multi-tasking matching model
[[Paper](https://dl.acm.org/doi/10.1145/3665252.3665263)]

2. Symphony: Towards Natural Language Query Answering over Multi-modal Data Lakes. [[Paper](https://www.vldb.org/cidrdb/2023/symphony-towards-natural-language-query-answering-over-multi-modal-data-lakes.html)]

#### 5.4.2 LLM for Heterogeneous Data Retrieval

1. Lotus: Enabling semantic queries with llms over tables of unstructured and structured data [[Paper](https://arxiv.org/html/2407.11418v1)]

2. Towards Operationalizing Heterogeneous Data Discovery [[Paper](https://arxiv.org/abs/2504.02059)]

3. CAESURA: Language Models as Multi-Modal Query Planners [[Paper](https://arxiv.org/abs/2308.03424)]

#### 5.4.2 Heterogeneous Data Analysis Agents

1. Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent [[Paper](https://arxiv.org/abs/2412.18428)]

2. An Interactive Multi-modal Query Answering System with Retrieval-Augmented Large Language Models [[Paper](https://arxiv.org/abs/2407.04217)]


3. Must: An effective and scalable framework for multimodal search of target modality [[Paper](https://arxiv.org/abs/2312.06397)]
