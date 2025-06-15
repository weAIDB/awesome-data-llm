# A Survey of LLM × DATA



> A collection of papers and projects related to LLMs and corresponding data-centric methods. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://arxiv.org/abs/2505.18458)
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
```



## 🌤 The IaaS Concept of DATA4LLM



The **IaaS** concept for LLM data (phonetically echoing *Infrastructure as a Service*) defines the characteristics of high-quality datasets along four key dimensions: (1) **Inclusiveness** ensures broad coverage across domains, tasks, sources, languages, styles, and modalities. (2) **Abundance** emphasizes sufficient and well-balanced data volume to support scaling, fine-tuning, and continual learning without overfitting. (3) **Articulation** requires clear, coherent, and instructive content with step-by-step reasoning to enhance model understanding and task performance. (4) **Sanitization** involves rigorous filtering to remove private, toxic, unethical, and misleading content, ensuring data safety, neutrality, and compliance. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/data_llm_survey_v3.pdf)

[![Cover](https://github.com/SUPERZJ827/LLM4DB/raw/main/assets/iaas_overview_v2.png)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/iaas_overview_v2.png)



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
  - [4.2 LLM for Data Analysis](#42-llm-for-data-analysis)
  - [4.3 LLM for Data System Optimization](#43-llm-for-data-system-optimization)


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
   Yukun Zhu, Ryan Kiros, Richard Zemel, et al. *ICCV 2015*.[[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)] 

### Data for Continual Pre-training

1. **MedicalGPT: Training Medical GPT Model**   
   Ming Xu. [[Github](https://github.com/shibing624/MedicalGPT)]
2. **BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark**   
   Dakuan Lu, Hengkui Wu, Jiaqing Liang, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2302.09432 )] 

### Data for Supervised Fine-Tuning (SFT)

#### General Instruction Following

1. **Free dolly: Introducing the world’s first truly open instruction-tuned llm**    
Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin. [[Source](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)]

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
   Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, Vicente Grau. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2408.04187)]
3. **ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization**  
   Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, Min Xu. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.06683)]
4. **PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents**  
   Saber Zerhoudi, Michael Granitzer. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.09394)]
5. **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services** [[Paper](https://arxiv.org/abs/2309.11325)]

### Data for LLM Evaluation

1. **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**  
   Xiang Yue, Yuansheng Ni, Kai Zhang, et al. *CVPR 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)]
2. **LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models**  
   Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, Yiqun Liu. *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20288)]
3. **What disease does this patient have? a large-scale open domain question answering dataset from medical exams**   
   Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits. *AAAI 2021*. [[Paper](https://arxiv.org/abs/2009.13081)]
4. **Evaluating Large Language Models Trained on Code**  
   Mark Chen, Jerry Tworek, Heewoo Jun, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2107.03374)]

### Data for LLM Agents

1. **STeCa: Step-level Trajectory Calibration for LLM Agent Learning**  
   Hanlin Wang, Jian Wang, Chak Tou Leong, Wenjie Li. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.14276)]
2. **Large Language Model-Based Agents for Software Engineering: A Survey**  
   Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, Yiling Lou. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.02977)]
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
     Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[Paper](https://arxiv.org/abs/1910.10683)]
10. **CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**  
    Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt. *arXiv 2019*. [[Paper](https://arxiv.org/abs/1909.09436)]
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
   Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. *NeurIPS 2024.* [[Paper](https://arxiv.org/abs/2405.14458)]
3. **UMIE: Unified Multimodal Information Extraction with Instruction Tuning**  
   Lin Sun, Kai Zhang, Qingyuan Li, Renze Lou. *AAAI 2024.* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29873)]
4. **ChatEL: Entity linking with chatbots**  
   Yifan Ding, Qingkai Zeng, Tim Weninger. *LREC | COLING 2024*. [[Paper](https://aclanthology.org/2024.lrec-main.275/)]
5. **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**  
   Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *ECCV 2024.* [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73235-5_23)]
6. **General OCR Theory: Towards OCR - 2.0 via a Unified End - to - end Model**  
   Haoran Wei, Chenglong Liu, Jinyue Chen, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2409.01704v1)]
7. **Focus Anywhere for Fine-grained Multi-page Document Understanding**  
   Chenglong Liu, Haoran Wei, Jinyue Chen, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14295)]
8. **MinerU: An Open-Source Solution for Precise Document Content Extraction**  
   Bin Wang, Chao Xu, Xiaomeng Zhao, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2409.18839)]
9. **WebIE: Faithful and Robust Information Extraction on the Web**  
   Chenxi Whitehouse, Clara Vania, Alham Fikri Aji, Christos Christodoulopoulos, Andrea Pierleoni. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-long.428/)]
10. **ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking**  
    Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, Andrea Pierleoni. *NAACL 2022 Industry Track.* [[Paper](https://aclanthology.org/2022.naacl-industry.24.pdf)]
11. **Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction**  
    Keshav Kolluru, Muqeeth Mohammed, Shubham Mittal, Soumen Chakrabarti, Mausam. *ACL 2022.* [[Paper](https://aclanthology.org/2022.acl-long.179/)]
12. **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**  
    Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. *ACM Multimedia 2022.* [[Paper](https://arxiv.org/abs/2204.08387)]
13. **Learning Transferable Visual Models From Natural Language Supervision**  
    Alec Radford, Jong Wook Kim, Chris Hallacy, et al. *ICML 2021.* [[Paper](https://proceedings.mlr.press/v139/radford21a)]
14. **Tesseract: an open-source optical character recognition engine**  
    Anthony Kay. Linux Journal, Volume 2007. [[Paper](https://dl.acm.org/doi/10.5555/1288165.1288167)]



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
   Fenghao Dong, Yang He, Yutong Liang, Zirui Liu, Yuhan Wu, Peiqing Chen, Tong Yang. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2405.19711)] 
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
   Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *ICLR 2023.* [[Paper](https://iclr.cc/virtual/2023/13610)]
5. **OPT: Open Pre-trained Transformer Language Models**  
   Susan Zhang, Stephen Roller, Naman Goyal, et al. *arXiv 2022.* [[Paper](https://arxiv.org/abs/2205.01068v4)]
6. **Learning Transferable Visual Models From Natural Language Supervision** [[Paper](https://proceedings.mlr.press/v139/radford21a)]
7. **OpenCLIP**     
   Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, et al. *2021*. [[Paper](https://doi.org/10.5281/zenodo.5143773)]
8. **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs**  
   Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, Aran Komatsuzaki. *NeurIPS 2021.* [[Paper](https://doi.org/10.48550/arXiv.2111.02114)]

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
   Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, Mansheej Paul. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/31214)]
2. **Data-efficient Fine-tuning for LLM-based Recommendation**  
   Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua. *SIGIR 2024.* [[Paper](https://arxiv.org/abs/2401.17197)]
3. **SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning**  
   Yexiao He, Ziyao Wang, Zheyu Shen, Guoheng Sun, Yucong Dai, Yongkai Wu, Hongyi Wang, Ang Li. *NeurIPS 2024.* [[Paper](https://arxiv.org/abs/2405.00705)]
4. **SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models**  
   Yu Yang, Siddhartha Mishra, Jeffrey Chiang, Baharan Mirzasoleiman. *NeurIPS 2024.* [[Paper](https://neurips.cc/virtual/2024/poster/95679)]
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
    Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2304.03277)]
14. **DINOv2: Learning Robust Visual Features without Supervision**  
    Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2304.07193)]
15. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**  
    Leo Gao, Stella Biderman, Sid Black, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2101.00027)]
16. **Language Models are Unsupervised Multitask Learners**  
    Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. *OpenAI blog 2019*. [[Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)]
17. **Bag of Tricks for Efficient Text Classification**  
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *EACL 2017.* [[Paper](https://aclanthology.org/E17-2068.pdf)]
18. **The Shapley Value: Essays in Honor of Lloyd S. Shapley**  
    A. E. Roth, Ed. *Cambridge: Cambridge University Press, 1988*. [[Source](https://www.cambridge.org/core/books/shapley-value/D3829B63B5C3108EFB62C4009E2B966E)]

##### (2) Model Scoring

1. **SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection**  
   Han Shen, Pin-Yu Chen, Payel Das, Tianyi Chen. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/29422)]
2. **QuRating: Selecting High-Quality Data for Training Language Models**  
   Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen. *ICML 2024.* [[Paper](https://arxiv.org/abs/2402.09739)]
3. **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**  
   Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He. *ICLR 2024.* [[Paper](https://arxiv.org/abs/2312.15685)]
4. **LAB: Large-Scale Alignment for ChatBots**  
   Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.01081)]
5. **Biases in Large Language Models: Origins, Inventory, and Discussion**  
   Roberto Navigli, Simone Conia, Björn Ross. *ACM JDIQ, 2023.* [[Paper](https://doi.org/10.1145/3597307)]

##### (3) Hybrid Methods

1. **Emergent and predictable memorization in large language models**  
   Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, et al. *NeurIPS 2023*. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667341?__cf_chl_tk=sWnInkGSOKRsrS.z3RwRKDT836eoSy1i.k5oxZcfDzA-1748509375-1.0.1.1-lmH0EWkZpuiyEr5uZPEd_C92GFkM6u6BY416q24qBww)]
2. **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**  
   Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, Sara Hooker. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.04564)]
3. **Instruction Mining: Instruction Data Selection for Tuning Large Language Models**  
   Yihan Cao, Yanbin Kang, Chi Wang, Lichao Sun. *arxiv 2023.* [[Paper](https://arxiv.org/abs/2307.06290)]
4. **Llama 2: Open Foundation and Fine-Tuned Chat Models**  
   Hugo Touvron, Louis Martin, Kevin Stone, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2307.09288)]
5. **MoDS: Model-oriented Data Selection for Instruction Tuning**  
   Qianlong Du, Chengqing Zong, Jiajun Zhang. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2311.15653)]
6. **Economic Hyperparameter Optimization With Blended Search Strategy**  
   Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. *ICLR 2021.* [[Paper](https://iclr.cc/virtual/2021/poster/3052)]
7. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *NAACL 2019.* [[Paper](https://aclanthology.org/N19-1423.pdf)]
8. **Active Learning for Convolutional Neural Networks: A Core-Set Approach**  
   Ozan Sener, Silvio Savarese. *ICLR 2018.* [[Paper](https://doi.org/10.48550/arXiv.1708.00489)]

#### Content-level Filtering

1. **spaCy**: An industrial-strength Natural Language Processing (NLP) library that supports tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and more; well-suited for fast and accurate text processing and information extraction. [[Source](https://spacy.io/)]
2. **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**  
   Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. ICLR 2025. [[Paper](https://arxiv.org/abs/2408.06072)]
3. **HunyuanVideo: A Systematic Framework For Large Video Generative Models**  
   Weijie Kong, Qi Tian, Zijian Zhang, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2412.03603v6)]
4. **Wan: Open and Advanced Large-Scale Video Generative Models**  
   Team Wan et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.20314)]
5. **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**   
   Hang Zhang, Xin Li, Lidong Bing. *EMNLP 2023 (System Demonstrations)*. [[Paper](https://arxiv.org/abs/2306.02858)]
6. **Analyzing Leakage of Personally Identifiable Information in Language Models**  
   Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin. *IEEE S&P 2023.* [[Paper](https://arxiv.org/abs/2302.00539)]
7. **DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4**  
   Zhengliang Liu, Yue Huang, Xiaowei Yu, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2303.11032)]
8. **Baichuan 2: Open Large-scale Language Models**  
   Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.10305)]
9. **Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives**  
   Haoning Wu, Erli Zhang, Liang Liao, et al. *arXiv 2022*. [[Paper](https://arxiv.org/abs/2211.04894)]
10. **YOLOX: Exceeding YOLO Series in 2021**  
      Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2107.08430)]
11. **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs** [[Paper](https://doi.org/10.48550/arXiv.2111.02114)]
12. **FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP**  
    Alan Akbik, Tanja Bergmann, Duncan Blythe, et al. *NAACL 2019 Demos.* [[Paper](https://aclanthology.org/N19-4010/)]

### 1.4 Data Selection

[⬆️top](#table-of-contents)

1. **A Survey on Data Selection for Language Models**  
   Alon Albalak, Yanai Elazar, Sang Michael Xie, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2402.16827)]

2. **A Survey on Data Selection for LLM Instruction Tuning**  
   Jiahao Wang, Bolin Zhang, Qianlong Du, Jiajun Zhang, Dianhui Chu. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2402.05123)]

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
   Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.04333)]
3. **TSDS: Data Selection for Task-Specific Model Finetuning**  
   Zifan Liu, Amin Karbasi, Theodoros Rekatsinas. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.11303)]
4. **Datamodels: Understanding Predictions with Data and Data with Predictions**  
   Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. *ICML 2022.* [[Paper](https://proceedings.mlr.press/v162/ilyas22a.html)]

#### Model-based Data Selection

1. **Autonomous Data Selection with Language Models for Mathematical Texts**  
   Yifan Zhang, Yifan Luo, Yang Yuan, Andrew Chi-Chih Yao. *ICLR 2024.* [[Paper](https://iclr.cc/virtual/2024/22423)]



### 1.5 Data Mixing

[⬆️top](#table-of-contents)

1. **Scalable Data Ablation Approximations for Language Models through Modular Training and Merging**  
   Clara Na, Ian Magnusson, Ananya Harsh Jha, et al. *EMNLP 2024.* [[Paper](https://arxiv.org/abs/2410.15661v1)]
2. **Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models**  
   Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, Hao Wang. *COLING 2024.* [[Paper](https://arxiv.org/abs/2403.03432v1)]

#### Heuristic Optimization

1. **BiMix: Bivariate Data Mixing Law for Language Model Pretraining**  
   Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, Bolin Ding. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14908)]
2. **Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining**  
   Steven Feng, Shrimai Prabhumoye, Kezhi Kong, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.15285)]
3. **SlimPajama-DC: Understanding Data Combinations for LLM Training** [[Paper](https://arxiv.org/abs/2309.10818)]
4. **Evaluating Large Language Models Trained on Code** [[Paper](https://arxiv.org/abs/2107.03374)]
5. **Exploring the limits of transfer learning with a unified text-to-text transformer** [[Paper](https://arxiv.org/abs/1910.10683v4)]
6. **CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge**  
   Alon Talmor, Jonathan Herzig, Nicholas Lourie, Jonathan Berant. *NAACL 2019*. [[Paper](https://arxiv.org/abs/1811.00937)]
7. **A mathematical theory of communication**  
   C. E. Shannon. *The Bell system technical journal 1948*. [[Paper](https://ieeexplore.ieee.org/document/6773024)]

#### Bilevel Optimization

1. **ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting**  
   Rui Pan, Jipeng Zhang, Xingyuan Pan, Renjie Pi, Xiaoyu Wang, Tong Zhang. *ACL 2025.* [[Paper](https://arxiv.org/abs/2406.19976)]
2. **DoGE: Domain Reweighting with Generalization Estimation**  
   Simin Fan, Matteo Pagliardini, Martin Jaggi. *ICML 2024.* [[Paper](https://icml.cc/virtual/2024/poster/34869)]
3. **An overview of bilevel optimization**  
   Benoît Colson, Patrice Marcotte, Gilles Savard. *AOR 2007.* [[Paper](https://link.springer.com/article/10.1007/s10479-007-0176-2)]

#### Distributionally Robust Optimization

1. **Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval**  
   Guangyuan Ma, Yongliang Ma, Xing Wu, Zhenpeng Su, Ming Zhou, Songlin Hu. *AAAI 2025.* [[Paper](https://arxiv.org/abs/2408.10613)]
2. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**  
   Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. *NeurIPS 2023.* [[Paper](https://arxiv.org/abs/2305.10429)]
3. **Qwen Technical Report**  
   Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, et al. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2309.16609v1)]

#### Model-Based Optimization

1. **RegMix: Data Mixture as Regression for Language Model Pre-training**  
   Qian Liu, Xiaosen Zheng, Niklas Muennighoff, et al. *ICLR 2025.* [[Paper](https://iclr.cc/virtual/2025/poster/30960)]
2. **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**  
   Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, Xipeng Qiu. *ICLR 2025.* [[Paper](https://arxiv.org/abs/2403.16952)]
3. **CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models**  
   Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, Fei Tan. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.903)]
4. **TinyLlama: An Open-Source Small Language Model**  
   Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2401.02385)]
5. **BiMix: Bivariate Data Mixing Law for Language Model Pretraining** [[Paper](https://arxiv.org/abs/2405.14908)]
6. **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models**  
   Haoran Que, Jiaheng Liu, Ge Zhang, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.01375)]
7. **Data Proportion Detection for Optimized Data Management for Large Language Models**  
   Hao Liang, Keshi Zhao, Yajie Yang, Bin Cui, Guosheng Dong, Zenan Zhou, Wentao Zhang. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2409.17527)]
8. **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining** [[Paper](https://arxiv.org/abs/2305.10429)]
9. **Training compute-optimal large language models**  
   Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. *NeurIPS 2022.* [[Paper](https://dl.acm.org/doi/10.5555/3600270.3602446)]
10. **LightGBM: a highly efficient gradient boosting decision tree**  
    Guolin Ke, Qi Meng, Thomas Finley, et al. *NeurIPS 2017.* [[Paper](https://dl.acm.org/doi/10.5555/3294996.3295074)]



### 1.6 Data Distillation and Synthesis

[⬆️top](#table-of-contents)

1. **How to Synthesize Text Data without Model Collapse?**  
   Xuekai Zhu, Daixuan Cheng, Hengli Li, et al. *ICML 2025*. [[Paper](https://arxiv.org/abs/2412.14689)]
2. **Differentially Private Synthetic Data via Foundation Model APIs 2: Text**  
   Chulin Xie, Zinan Lin, Arturs Backurs, et al. *ICML 2024.* [[Paper](https://arxiv.org/abs/2403.01749v2)]
3. **LLM See, LLM Do: Leveraging Active Inheritance to Target Non-Differentiable Objectives**  
   Luísa Shimabucoro, Sebastian Ruder, Julia Kreutzer, Marzieh Fadaee, Sara Hooker. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.521)]
4. **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions** [[Paper](https://iclr.cc/virtual/2024/poster/19164)]
5. **Augmenting Math Word Problems via Iterative Question Composing**  
   Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.09003)]

#### Knowledge Distillation

1. **Multistage Collaborative Knowledge Distillation from a Large Language Model for Semi-Supervised Sequence Generation**   
   Jiachen Zhao, Wenlong Zhao, Andrew Drozdov, et al. *ACL 2024*. [[Paper](https://arxiv.org/abs/2311.08640)]
2. **PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning**  
   Xuekai Zhu, Biqing Qi, Kaiyan Zhang, Xinwei Long, Zhouhan Lin, Bowen Zhou. *NAACL 2024*. [[Paper](https://arxiv.org/abs/2305.13888)]
3. **Knowledge Distillation Using Frontier Open-source LLMs: Generalizability and the Role of Synthetic Data**   
   Anup Shirgaonkar, Nikhil Pandey, Nazmiye Ceren Abay, Tolga Aktas, Vijay Aski. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.18588)]
4. **Training Verifiers to Solve Math Word Problems**  
   Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. *arXiv 2021*. [[Paper](https://arxiv.org/abs/2110.14168)]

#### Pre-training Data Augmentation

1. **BERT-Tiny-Chinese**: A lightweight Chinese BERT pre-trained model released by CKIP Lab, with a small number of parameters; suitable for use as an encoder in pre-training data augmentation tasks to enhance efficiency for compact models. [[Source](https://huggingface.co/ckiplab/bert-tiny-chinese)]
2. **Case2Code: Scalable Synthetic Data for Code Generation**   
   Yunfan Shao, Linyang Li, Yichuan Ma, et al. *COLING 2025*. [[Paper](https://aclanthology.org/2025.coling-main.733/)]
3. **Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages**  
   Zui Chen, Tianqiao Liu, Mi Tian, Qing Tong, Weiqi Luo, Zitao Liu. *ICLR 2025*. [[Paper](https://arxiv.org/abs/2501.14002)]
4. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**  
   Kun Zhou, Beichen Zhang, Jiapeng Wang, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2405.14365)]
5. **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**  
   Bin Xiao, Haiping Wu, Weijian Xu, et al. *CVPR 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)]
6. **DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models**  
   Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar. *CVPR 2024*. [[Paper](https://arxiv.org/abs/2405.14881)]
7. **Magicoder: Empowering Code Generation with OSS-Instruct**   
   Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang. *ICML 2024*. [[Paper](https://arxiv.org/abs/2312.02120)]
8. **Instruction Pre-Training: Language Models are Supervised Multitask Learners**  
   Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, Furu Wei. *EMNLP 2024*. [[Paper](https://arxiv.org/abs/2406.14491)]
9. **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research** [[Paper](https://arxiv.org/abs/2402.00159)]
10. **Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling**   
    Pratyush Maini, Skyler Seto, Richard Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly. *ACL 2024*. [[Paper](https://aclanthology.org/2024.acl-long.757/)]
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
    Hao Chen, Abdul Waheed, Xiang Li, Yidong Wang, Jindong Wang, Bhiksha Raj, Marah I. Abdin. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2410.15226)]
17. **Towards Effective and Efficient Continual Pre-training of Large Language Models**  
    Jie Chen, Zhipeng Chen, Jiapeng Wang, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.18743)]
18. **Improving CLIP Training with Language Rewrites**  
    Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian. *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2305.20088)]
19. **Effective Data Augmentation With Diffusion Models**  
    Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2302.07944)]
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
    Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. *Findings of EMNLP 2020*. [[Paper](https://arxiv.org/abs/1909.10351)]
27. **HellaSwag: Can a Machine Really Finish Your Sentence?**  
    Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. *ACL 2019*. [[Paper](https://arxiv.org/abs/1905.07830)]

#### SFT Data Augmentation

1. **Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning**  
   Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.02333)]
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
   Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.03387)]
3. **LLMs Can Easily Learn to Reason from Demonstrations: Structure, Not Content, Is What Matters!**  
   Dacheng Li, Shiyi Cao, Tyler Griggs, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.07374)]
4. **Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search**  
   Maohao Shen, Guangtao Zeng, Zhenting Qi, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.02508)]
5. **Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling**  
   Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, Yuxiao Dong. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.11651)]
6. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data**  
   Yinya Huang, Xiaohan Lin, Zhengying Liu, et al. *ICLR 2024.* [[Paper](https://arxiv.org/abs/2402.08957v3)]
7. **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**  
   Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, Zhifang Sui. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.510)]
8. **NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions**   
   Jia Li, Edward Beeching, Lewis Tunstall, et al. *2024*. [[Paper](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)]
9. **QwQ: Reflect Deeply on the Boundaries of the Unknown**   
   Qwen Team. *2024*. [[Source](https://qwenlm.github.io/blog/qwq-32b-preview/)]
10. **Let's Verify Step by Step**  
    Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2305.20050)]

#### Reinforcement Learning

1. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
   Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. NeurIPS 2023. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3668142)]
2. **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**  
   Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, et al. *arXiv 2022.* [[Paper](https://doi.org/10.48550/arXiv.2204.05862)]

#### Retrieval-Augmentation Generation

1. **Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data**  
   Shenglai Zeng, Jiankun Zhang, Pengfei He, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.14773)]



### 1.7 End-to-End Data Processing Pipelines

[⬆️top](#table-of-contents)

#### 1.7.1 Typical data processing frameworks

1. **Data-Juicer: A One-Stop Data Processing System for Large Language Models**  
   Daoyuan Chen, Yilun Huang, Zhijian Ma, et al. *SIGMOD 2024.* [[Paper](https://doi.org/10.1145/3626246.3653385)]
2. **An Integrated Data Processing Framework for Pretraining Foundation Models**  
   Yiding Sun, Feng Wang, Yutao Zhu, Wayne Xin Zhao, Jiaxin Mao. *SIGIR 2024.* [[Paper](https://doi.org/10.1145/3626772.3657671)]
3. **Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models**  
   Hyunbyung Park, Sukyung Lee, Gyoungjin Gim, Yungi Kim, Dahyun Kim, Chanjun Park. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2403.19340v1)]

#### 1.7.2 Typical data pipelines

1. **Common Crawl**: A large-scale publicly accessible web crawl dataset that provides massive raw webpages and metadata. It serves as a crucial raw data source in typical pretraining data pipelines, where it undergoes multiple processing steps such as cleaning, deduplication, and formatting to produce high-quality corpora for downstream model training. [[Source](https://commoncrawl.org/)]
2. **The RefinedWeb dataset for falcon LLM: outperforming curated corpora with web data only**  
   Guilherme Penedo, Quentin Malartic, Daniel Hesslow, et al. *NeurIPS 2023.* [[Paper](https://dl.acm.org/doi/10.5555/3666122.3669586)]
3. **Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction** [[Paper](https://aclanthology.org/2021.acl-demo.15.pdf)]
4. **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**  
   Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, et al. *arXiv 2021.* [[Paper](https://arxiv.org/abs/2112.11446v2)]
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
   Beatrice Casey, Kaia Damian, Andrew Cotaj, Joanna C. S. Santos. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.02170)]



### 2.2 Data Distribution

[⬆️top](#table-of-contents)

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** [[Paper](https://arxiv.org/abs/2501.12948)]
2. **CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl**  
   Ilya Ilyankou, Meihui Wang, Stefano Cavazzi, James Haworth. *SIGSPATIAL 2024.* [[Paper](https://doi.org/10.1145/3678717.3691215)]

#### Distributed Storage Systems

1. **JuiceFS**: A high-performance cloud-native distributed file system designed for efficient storage and access of large-scale data. [[Github](https://github.com/juicedata/juicefs)]
2. **3FS**: A distributed file system designed for deep learning and large-scale data processing, emphasizing high throughput and reliability. [[Github](https://github.com/deepseek-ai/3fs)]
3. **S3**: A widely used cloud storage service offering secure, scalable, and highly available object storage solutions. [[Source](https://aws.amazon.com/s3)]
4. **Hdfs architecture guide. Hadoop apache project**  
    D. Borthakur et al. *Hadoop apache project, 53(1-13):2, 2008*.[[Source](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)]

#### Heterogeneous Storage Systems

1. **ProTrain: Efficient LLM Training via Memory-Aware Techniques**  
   Hanmei Yang, Jin Zhou, Yao Fu, Xiaoqun Wang, Ramine Roane, Hui Guan, Tongping Liu. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2406.08334)]
2. **ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning**  
   Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. *SC 2021.* [[Paper](https://doi.org/10.1145/3458817.3476205)]
3. **ZeRO-Offload: Democratizing Billion-Scale Model Training**  
   Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, et al. *USENIX ATC 2021.* [[Paper](https://www.usenix.org/system/files/atc21-ren-jie.pdf)]
4. **ZeRO: memory optimizations toward training trillion parameter models**  
   Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *SC 2020.* [[Paper](https://dl.acm.org/doi/10.5555/3433701.3433727)]
5. **vDNN: virtualized deep neural networks for scalable, memory-efficient neural network design**  
   Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, Stephen W. Keckler. *MICRO-49 2016.* [[Paper](https://dl.acm.org/doi/10.5555/3195638.3195660)]



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
   Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, Zengchang Qin. *COLING 2025.* [[Paper](https://doi.org/10.48550/arXiv.2406.00456)]
6. **Dense X Retrieval: What Retrieval Granularity Should We Use?**  
   Tong Chen, Hongwei Wang, Sihao Chen, et al. *EMNLP 2024*. [[Paper](https://aclanthology.org/2024.emnlp-main.845/)]
7. **Scalable and Domain-General Abstractive Proposition Segmentation**  
   Mohammad Javad Hosseini, Yang Gao, Tim Baumgärtner, et al. *Findings of EMNLP 2024*. [[Paper](https://aclanthology.org/2024.findings-emnlp.517/)]
8. **A Hierarchical Context Augmentation Method to Improve Retrieval-Augmented LLMs on Scientific Papers**  
   Tian-Yi Che, Xian-Ling Mao, Tian Lan, Heyan Huang. *KDD 2024*. [[Paper](https://dl.acm.org/doi/10.1145/3637528.3671847)]
9. **M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**  
   Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.137.pdf)]
10. **Thread: A Logic-Based Data Organization Paradigm for How-To Question Answering with Retrieval Augmented Generation**  
    Kaikai An, Fangkai Yang, Liqun Li, et al. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2406.13372)]
11. **GleanVec: Accelerating Vector Search with Minimalist Nonlinear Dimensionality Reduction**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Ted Willke. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2410.22347)]
12. **The Faiss Library**  
    Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.08281)]
13. **Similarity Search in the Blink of an Eye with Compressed Indices**  
    Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, et al. *VLDB Endowment 2023.* [[Paper](https://doi.org/10.14778/3611479.3611537)]
14. **LeanVec: Searching Vectors Faster by Making Them Fit**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Mark Hildebrand, Ted Willke. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2312.16335)]
15. **Towards General Text Embeddings with Multi-stage Contrastive Learning**  
    Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2308.03281)]

#### Graph-Based Organization

1. **ArangoDB**: A multi-model database that supports graph, document, and key-value data, suitable for handling complex relational queries. [[Source](https://arangodb.com/)]
2. **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**  
   Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.06713)]
3. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**  
   Darren Edge, Ha Trinh, Newman Cheng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.16130)]
4. **LightRAG: Simple and Fast Retrieval-Augmented Generation**  
   Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.05779)]
5. **Graph Databases Assessment: JanusGraph, Neo4j, and TigerGraph**  
   Jéssica Monteiro, Filipe Sá, Jorge Bernardino. *Perspectives and Trends in Education and Technology 2023.* [[Paper](https://doi.org/10.1007/978-981-19-6585-2_58)]
6. **Empirical Evaluation of a Cloud-Based Graph Database: the Case of Neptune**  
   Ghislain Auguste Atemezing. *KGSWC 2021.* [[Paper](https://doi.org/10.1007/978-3-030-91305-2_3)]



### 2.4 Data Movement

[⬆️top](#table-of-contents)

#### Caching Data

1. **CacheLib**: An open-source, high-performance embedded caching library developed by Meta to accelerate data access and increase system throughput. [[Source](https://cachelib.org/)]
2. **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**  
   Mark Zhao, Satadru Pan, Niket Agarwal, et al. *USENIX ATC 2023.* [[Paper](https://www.usenix.org/conference/atc23/presentation/zhao)]
3. **Fluid: Dataset Abstraction and Elastic Acceleration for Cloud-native Deep Learning Training Jobs**  
   Rong Gu, Kai Zhang, Zhihao Xu, Yang Che, Bin Fan, Haojun Hou. *ICDE 2022.* [[Paper](https://doi.org/10.1109/ICDE53745.2022.00209)]
4. **Quiver: An Informed Storage Cache for Deep Learning**  
   Abhishek Kumar, Muthian Sivathanu. *USENIX FAST 2020.* [[Paper](https://www.usenix.org/conference/fast20/presentation/kumar)]

#### Data/Operator Offloading

1. **cedar: Optimized and Unified Machine Learning Input Data Pipelines**  
   Mark Zhao, Emanuel Adamiak, Christos Kozyrakis. *Proceedings of the VLDB Endowment, Volume 18, Issue 2, 2025.* [[Paper](https://dl.acm.org/doi/10.14778/3705829.3705861)]
2. **Pecan: cost-efficient ML data preprocessing with automatic transformation ordering and hybrid placement**  
   Dan Graur, Oto Mraz, Muyu Li, Sepehr Pourghannad, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2024.* [[Paper](https://dl.acm.org/doi/10.5555/3691992.3692032)]
3. **tf.data service: A Case for Disaggregating ML Input Data Processing**  
   Andrew Audibert, Yang Chen, Dan Graur, Ana Klimovic, Jiří Šimša, Chandramohan A. Thekkath. *SoCC 2023.* [[Paper](https://doi.org/10.1145/3620678.3624666)]
4. **Cachew: Machine Learning Input Data Processing as a Service**  
   Dan Graur, Damien Aymon, Dan Kluser, Tanguy Albrici, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2022.* [[Paper](https://www.usenix.org/conference/atc22/presentation/graur)]
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
   Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, et al. *USENIX NSDI 2024.* [[Paper](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)]
3. **ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development**  
   Borui Wan, Mingji Han, Yiyao Sheng, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2407.20143)]
4. **GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints**  
   Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. *SOSP 2023.* [[Paper](https://doi.org/10.1145/3600006.3613145)]
5. **CheckFreq: Frequent, Fine-Grained DNN Checkpointing**  
   Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *USENIX FAST 2021.* [[Paper](https://www.usenix.org/conference/fast21/presentation/mohan)]

#### Redundant Computations

1. **ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation**  
   Swapnil Gandhi, Mark Zhao, Athinagoras Skiadopoulos, Christos Kozyrakis. *SOSP 2024*. [[Paper](https://arxiv.org/abs/2405.14009)]
2. **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs**  
   John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, et al.  *NSDI 2023* . [[Paper](https://www.usenix.org/conference/nsdi23/presentation/thorpe)]
3. **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates**  
   Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, Mosharaf Chowdhury. *SOSP 2023*. [[Paper](https://dl.acm.org/doi/10.1145/3600006.3613152)]



### 2.6 KV Cache

[⬆️top](#table-of-contents)

#### Cache Space Management

1. **Efficient Memory Management for Large Language Model Serving with PagedAttention**  
   Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. *SOSP 2023.* [[Paper](https://arxiv.org/abs/2309.06180)]
2. **VTensor: Using Virtual Tensors to Build a Layout-oblivious AI Programming Framework**  
   Feng Yu, Jiacheng Zhao, Huimin Cui, Xiaobing Feng, Jingling Xue. *PACT 2020.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3410463.3414664)]

#### KV Placement

1. **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**  
   Bin Gao, Zhuomin He, Puru Sharma, et al. *USENIX ATC 2024.* [[Paper](https://arxiv.org/abs/2403.19708)]
2. **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**  
   Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2404.12457)]

#### KV Shrinking

1. **Fast State Restoration in LLM Serving with HCache**  
   Shiwei Gao, Youmin Chen, Jiwu Shu. *EuroSys 2025.* [[Paper](https://arxiv.org/abs/2410.05004)]
2. **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**  
   Yuhan Liu, Hanchen Li, Yihua Cheng, et al. *SIGCOMM 2024.* [[Paper](https://dl.acm.org/doi/abs/10.1145/3651890.3672274)]
3. **MiniCache: KV Cache Compression in Depth Dimension for Large Language Models**  
   Akide Liu · Jing Liu · Zizheng Pan · Yefei He · Reza Haffari · Bohan Zhuang. *NeurIPS 2024*. [[Paper](https://neurips.cc/virtual/2024/poster/93380)]
4. **Animating rotation with quaternion curves**  
   Ken Shoemake. *ACM SIGGRAPH Computer Graphics, Volume 19, Issue 3. 1985*. [[Paper](https://dl.acm.org/doi/10.1145/325165.325242)]

#### KV Indexing

1. **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**  
   Lu Ye, Ze Tao, Yong Huang, Yang Li. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.623/)]
2. **BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching**  
   Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.03594)]



## 3 Data Serving for LLM

[⬆️top](#table-of-contents)

### 3.1 Data Shuffling

#### Data Shuffling for Training

1. **Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training**  
   Zheheng Luo, Xin Zhang, Xiao Liu, Haoling Li, Yeyun Gong, Chen Qi, Peng Cheng. *ACL 2025*. [[Paper](https://arxiv.org/abs/2411.14318)]
2. **How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition**  
   Guanting Dong, Hongyi Yuan, Keming Lu, et al. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.12/)]
3. **Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models**    
   Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari. *EMNLP 2024.* [[Paper](https://aclanthology.org/2024.emnlp-main.787/)]
4. **Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning**  
   Jisu Kim, Juhwan Lee. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.07490)]
5. **NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks**  
   Jean-michel Attendu, Jean-philippe Corbeil. *SustaiNLP @ ACL 2023.* [[Paper](https://aclanthology.org/2023.sustainlp-1.9/)]
6. **Efficient Online Data Mixing For Language Model Pre-Training**  
   Alon Albalak, Liangming Pan, Colin Raffel, William Yang Wang. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2312.02406)]
7. **Data Pruning via Moving-one-Sample-out**  
   Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi. *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2310.14664)]
8. **BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning**  
   Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi, Mohammad Taher Pilehvar, Yadollah Yaghoobzadeh, Samira Ebrahimi Kahou. *ENLSP @ NeurIPS2022.* [[Paper](https://doi.org/10.48550/arXiv.2211.05610)]
9. **Scaling Laws for Neural Language Models**  
   Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. *arXiv 2020*. [[Paper](https://arxiv.org/abs/2001.08361)]
10. **Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory**  
    James L. McClelland, Bruce L. McNaughton, Randall C. O’Reilly. *Psychological Review 1995.* [[Paper](https://cseweb.ucsd.edu/~gary/258/jay.pdf)]
11. **Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem**  
    M. McCloskey, N. J. Cohen. *Psychology of Learning and Motivation 1989.* [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)]

#### Data Selection for RAG

1. **Cohere rerank**: Cohere's rerank model reorders initial retrieval results to improve relevance to the query, making it a key component for building high-quality RAG systems. [[Source](https://docs.cohere.com)]
2. **ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval**  
   Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt. *NAACL 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.15245)]
3. **MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation**  
   Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, et al. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2501.00332)]
4. **ARAGOG: Advanced RAG Output Grading**  
   Matouš Eibich, Shivay Nagpal, Alexander Fred-Ojala. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.01037)]
5. **Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!**  
   Yubo Ma, Yixin Cao, YongChing Hong, Aixin Sun. *Findings of EMNLP 2023*. [[Paper](https://aclanthology.org/2023.findings-emnlp.710/)]
6. **Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model**  
   Jiaxi Cui, Munan Ning, Zongjian Li, et al. *arXiv 2023*.[[Paper](https://arxiv.org/abs/2306.16092v2)]
7. **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**  
   Ronak Pradeep, Sahel Sharifymoghaddam, Jimmy Lin. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2309.15088)]



### 3.2 Data Compression

[⬆️top](#table-of-contents)

#### RAG Knowledge Compression

1. **Context Embeddings for Efficient Answer Generation in RAG**  
   David Rau, Shuai Wang, Hervé Déjean, Stéphane Clinchant. *WSDM 2025.* [[Paper](https://doi.org/10.48550/arXiv.2407.09252)]
2. **xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**  
   Xin Cheng, Xun Wang, Xingxing Zhang, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.13792)]
3. **RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation**  
   Fangyuan Xu, Weijia Shi, Eunsol Choi. *ICLR 2024.* [[Paper](https://iclr.cc/virtual/2024/poster/17885)]
4. **Compressing Long Context for Enhancing RAG with AMR-based Concept Distillation**   
   Kaize Shi, Xueyao Sun, Qing Li, Guandong Xu. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.03085)]
5. **Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation**  
   Dongwon Jung, Qin Liu, Tenghao Huang, Ben Zhou, Muhao Chen. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.12468)]

#### Prompt Compression

1. **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**  
   Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.91/)]
2. **LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression**  
   Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, et al. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.57/)]
3. **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**  
   Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *EMNLP 2023.* [[Paper](https://aclanthology.org/2023.emnlp-main.825.pdf)]
4. **Learning to Compress Prompts with Gist Tokens**  
   Jesse Mu, Xiang Lisa Li, Noah Goodman. *NeurIPS 2023.* [[Paper](https://arxiv.org/abs/2304.08467)]
5. **Adapting Language Models to Compress Contexts**  
   Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen. *EMNLP 2023.* [[Paper](https://aclanthology.org/2023.emnlp-main.232.pdf)]



### 3.3 Data Packing

[⬆️top](#table-of-contents)

#### Short Sequence Insertion

1. **Fewer Truncations Improve Language Modeling**  
   Hantian Ding, Zijian Wang, Giovanni Paolini, et al. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2404.10830)]
2. **Bucket Pre-training is All You Need**  
   Hongtao Liu, Qiyao Peng, Qing Yang, Kai Liu, Hongyan Xu. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2407.07495)]

#### Sequence Combination Optimization

1. **Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum**  
   Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.13226)]
2. **Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance**  
   Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon. *arXiv 2021.* [[Paper](https://doi.org/10.48550/arXiv.2107.02027)]

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
   Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren. *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2406.01946)]
2. **Undetectable Watermarks for Language Models**  
   Miranda Christ, Sam Gunn, Or Zamir. in *Proceedings of the 37th Annual Conference on Learning Theory (COLT 2024)*. [[Paper](https://arxiv.org/abs/2306.09194)]
3. **An Unforgeable Publicly Verifiable Watermark for Large Language Models**  
   Aiwei Liu, Leyi Pan, Xuming Hu, Shu'ang Li, Lijie Wen, Irwin King, Philip S. Yu. *ICLR 2024*. [[Paper](https://arxiv.org/abs/2307.16230)]
4. **A Watermark for Large Language Models**  
   John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *ICML 2023*. [[Paper](https://arxiv.org/abs/2301.10226)]
5. **Publicly-Detectable Watermarking for Language Models**   
   Jaiden Fairoze, Sanjam Garg, Somesh Jha, et al. *arXiv 2023*. [[Paper](https://arxiv.org/abs/2310.18491)]

#### Statistical Provenance

1. **A Watermark for Large Language Models** [[Paper](https://arxiv.org/abs/2301.10226)]



## 4 LLM for Data Management

[⬆️top](#table-of-contents)

### 4.1 LLM for Data Manipulation

#### 4.1.1 LLM for Data Cleaning

##### Data Standardization

1. **Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes**  
   Simran Arora, Brandon Yang, Sabri Eyuboglu, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 2, 2024.* [[Paper](https://dl.acm.org/doi/abs/10.14778/3626292.3626294)]
2. **CleanAgent: Automating Data Standardization with LLM-based Agents**  
   Danrui Qi, Jiannan Wang. *arXiv 2024.* [[Paper](https://arxiv.org/pdf/2403.08291)]
3. **AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark**  
   Lan Li, Liri Fang, Vetle I. Torvik. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.06724)]
4. **LLMs with User-defined Prompts as Generic Data Operators for Reliable Data Processing**  
   Luyi Ma, Nikhil Thakurdesai, Jiao Chen, et al. *1st IEEE International Workshop on Data Engineering and Modeling for AI (DEMAI), IEEE BigData 2023.* [[Paper](https://arxiv.org/abs/2312.16351)]

##### Data Error Processing

1. **GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models**  
   Mengyi Yan, Yaoshu Wang, Yue Wang, Xiaoye Miao, Jianxin Li. *Proceedings of the ACM on Management of Data, Volume 2, Issue 6, 2024.* [[Paper](https://dl.acm.org/doi/10.1145/3698811)]
2. **Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets**  
   Tommaso Bendinelli, Artur Dox, Christian Holz. *ICLR 2025 Workshop on Foundation Models in the Wild*. [[Paper](https://arxiv.org/abs/2503.06664)]
3. **Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation**  
   Juhwan Choi, Jungmin Yun, Kyohoon Jin, YoungBin Kim. *EMNLP 2024*. [[Paper](https://arxiv.org/abs/2404.09682)]
4. **Data Cleaning Using Large Language Models**  
   Shuo Zhang, Zezhou Huang, Eugene Wu. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.15547)]
5. **LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs**  
   Fabian Biester, Mohamed Abdelaal, Daniel Del Gaudio. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2404.18681)]

##### Data Imputation

1. **RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes**  
   Zan Ahmad Naeem, Mohammad Shahmeer Ahmad, Mohamed Eltabakh, et al. *VLDB Endowment 2024*. [[Paper](https://dl.acm.org/doi/10.14778/3685800.3685890)]



#### 4.1.2 LLM for Data Integration

##### Entity Matching

1. **Entity matching using large language models**  
   Ralph Peeters, Christian Bizer. *EDBT 2025.* [[Paper](https://arxiv.org/abs/2310.11244)]
2. **Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching**  
   Tianshu Wang, Hongyu Lin, Xiaoyang Chen, Xianpei Han, Hao Wang, Zhenyu Zeng, Le Sun. *COLING 2025.* [[Paper](https://aclanthology.org/2025.coling-main.8/)]
3. **Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration**  
   Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai, Nan Tang, Guoliang Li, Xiaoyong Du. *ICDE 2024.* [[Paper](https://ieeexplore.ieee.org/document/10597751)]
4. **KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs**  
   Yongqin Xu, Huan Li, Ke Chen, Lidan Shou. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2410.12480)]
5. **Jellyfish: A Large Language Model for Data Preprocessing**  
   Haochen Zhang, Yuyang Dong, Chuan Xiao, Masafumi Oyamada. *EMNLP 2024.* [[Paper](https://arxiv.org/abs/2312.01678)]

##### Schema Matching

1. **Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching**  
   Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, Bálint Molnár. *arxiv 2025.* [[Paper](https://arxiv.org/abs/2501.08686)]
2. **Interactive Data Harmonization with LLM Agents**  
   Aécio Santos, Eduardo H. M. Pena, Roque Lopez, Juliana Freire. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.07132)]
3. **Schema Matching with Large Language Models: an Experimental Study**  
   Marcel Parciak, Brecht Vandevoort, Frank Neven, et al. *TaDA 2024 Workshop, collocated with VLDB 2024.* [[Paper](https://doi.org/10.48550/arXiv.2407.11852)]
4. **Magneto: Combining Small and Large Language Models for Schema Matching**  
   Yurong Liu, Eduardo Pena, Aecio Santos, Eden Wu, Juliana Freire. *VLDB Endowment 2024.*  [[Paper](https://www.vldb.org/pvldb/vol17/p2750-fan.pdf)]
5. **Agent-OM: Leveraging LLM Agents for Ontology Matching**
   Zhangcheng Qiang, Weiqing Wang, Kerry Taylor. *Proceedings of the VLDB Endowment, Volume 18, Issue 3, 2024.* [[Paper](https://dl.acm.org/doi/10.14778/3712221.3712222)]



#### 4.1.3 LLM for Data Discovery

1. **ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models**  
   Benjamin Feuer, Yurong Liu, Chinmay Hegde, Juliana Freire. *VLDB 2024*. [[Paper](https://arxiv.org/abs/2310.18208#:~:text=We%20introduce%20ArcheType%2C%20a%20simple%2C%20practical%20method%20for,solve%20CTA%20problems%20in%20a%20fully%20zero-shot%20manner.)]

##### Data Profiling

1. **Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System**  
   Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, et al. *SIGMOD 2025*. [[Paper](https://arxiv.org/abs/2504.09207#:~:text=In%20this%20paper%2C%20we%20introduce%20Pneuma%2C%20a%20retrieval-augmented,designed%20to%20efficiently%20and%20effectively%20discover%20tabular%20data.)]
2. **AutoDDG: Automated Dataset Description Generation using Large Language Models**  
   Haoxiang Zhang, Yurong Liu, Wei-Lun (Allen) Hung, Aécio Santos, Juliana Freire. *arxiv 2025.* [[Paper](https://arxiv.org/abs/2502.01050)]
3. **LEDD: Large Language Model-Empowered Data Discovery in Data Lakes**  
   Qi An, Chihua Ying, Yuqing Zhu, Yihao Xu, Manwei Zhang, Jianmin Wang. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2502.15182)]

##### Data Annotation

1. **Birdie: Natural Language-Driven Table Discovery Using Differentiable Search Index**  
   Yuxiang Guo, Zhonghao Hu, Yuren Mao, Baihua Zheng, Yunjun Gao, Mingwei Zhou. *VLDB 2025*. [[Paper](https://arxiv.org/abs/2504.21282)]
2. **Mind the Data Gap: Bridging LLMs to Enterprise Data Integration**  
   Moe Kayali, Fabian Wenz, Nesime Tatbul, Çağatay Demiralp. *CIDR 2025.* [[Paper](https://arxiv.org/abs/2412.20331)]
3. **Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation**  
   Keti Korini, Christian Bizer. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2503.02718)]
4. **CHORUS: Foundation Models for Unified Data Discovery and Exploration**  
   Moe Kayali, Anton Lykov, Ilias Fountalis, Nikolaos Vasiloglou, Dan Olteanu, Dan Suciu. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[Paper](https://dl.acm.org/doi/10.14778/3659437.3659461)]
5. **RACOON: An LLM-based Framework for Retrieval-Augmented Column Type Annotation with a Knowledge Graph**  
   Lindsey Linxi Wei, Guorui Xiao, Magdalena Balazinska. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2409.14556)]



### 4.2 LLM for Data Analysis

[⬆️top](#table-of-contents)

#### 4.2.1 LLM for Structured Data Analysis

##### 4.2.1.1 Relational Data Analysis

###### LLM for Natural Language Interfaces

1. **Cracking SQL Barriers: An LLM-based Dialect Translation System**  
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *SIGMOD 2025*. [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/SIGMOD25-CrackSQL.pdf)]
2. **CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models**    
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *arXiv 2025*. [[Paper](https://arxiv.org/abs/2504.00882#:~:text=In%20this%20demonstration%2C%20we%20present%20CrackSQL%2C%20the%20first,rule%20and%20LLM-based%20methods%20to%20overcome%20these%20limitations.)]
3. **Automatic Metadata Extraction for Text-to-SQL**     
   Vladislav Shkapenyuk, Divesh Srivastava, Theodore Johnson, Parisa Ghane. *arXiv 2025* [[Paper](https://arxiv.org/abs/2505.19988)]
4. **CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning**     
   Lei Sheng, Shuai-Shuai Xu. *arXiv 2025* [[Paper](https://arxiv.org/abs/2505.13271)]
5. **Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL**       
   Lei Sheng, Shuai-Shuai Xu. *arXiv 2025* [[Paper](https://arxiv.org/abs/2505.13271)]
6. **OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale**    
   Haoyang Li, Shang Wu, Xiaokang Zhang, et al. *arXiv 2025* [[Paper](https://arxiv.org/abs/2503.02240)]
7. **OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment**   
   Xiangjin Xie, Guangwei Xu, Lingyan Zhao, Ruijie Guo. *arXiv 2025* [[Paper](https://arxiv.org/abs/2502.14913)]
8. **Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning**    
   Yusuf Denizay Dönder, Derek Hommel, Andrea W Wen-Yi, David Mimno, Unso Eun Seo Jo. *arXiv 2025* [[Paper](https://arxiv.org/abs/2505.14174)]
9. **A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL**   
   Yingqi Gao, Yifu Liu, Xiaoxia Li, et al. *arXiv 2025* [[Paper](https://arxiv.org/abs/2411.08599)]
10. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis**  
    Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin. *SIGMOD 2024.* [[Paper](https://doi.org/10.1145/3626246.3653375)]
11. **CodeS: Towards Building Open-source Language Models for Text-to-SQL**  
    Haoyang Li, Jing Zhang, Hanbing Liu, et al. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024.* [[Paper](https://doi.org/10.1145/3654930)]
12. **The Dawn of Natural Language to SQL: Are We Fully Ready?**  
    Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, Nan Tang. *VLDB 2024.* [[Paper](https://arxiv.org/abs/2406.01265)]
13. **Contextualized Data-Wrangling Code Generation in Computational Notebooks**  
    Junjie Huang, Daya Guo, Chenglong Wang, et al. *ASE 2024*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3691620.3695503)]
14. **PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**  
    Zhishuai Li, Xiang Wang, Jingjing Zhao, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.09732)]
15. **CHESS: Contextual Harnessing for Efficient SQL Synthesis**   
    Shayan Talaei, Mohammadreza Pourreza, Yu-Chen Chang, Azalia Mirhoseini, Amin Saberi. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2405.16755)]
16. **Data Interpreter: An LLM Agent For Data Science**  
    Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, et al. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2402.18679)]
17. **DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction**  
    Mohammadreza Pourreza, Davood Rafiei. *NeurIPS 2023*. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667699)]
18. **Natural Language to Code Generation in Interactive Data Science Notebooks**   
    Pengcheng Yin, Wen-Ding Li, Kefan Xiao, et al. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-long.9/)]
19. **PaLM: Scaling Language Modeling with Pathways**   
    Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. *JMLR 2023.* [[Paper](https://dl.acm.org/doi/10.5555/3648699.3648939)]

###### LLM for Semantic Analysis

1. **TableMaster: A Recipe to Advance Table Understanding with Language Models**  
   Lang Cao. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2501.19378)]
2. **RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals**  
   Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.15110)]
3. **PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models**  
   Wei Zhou, Mohsen Mesgar, Heike Adel, Annemarie Friedrich. *arXiv 2025.* [[Paper](https://arxiv.org/abs/2505.17565)]
4. **TAT-LLM: A Specialized Language Model for Discrete Reasoning over Financial Tabular and Textual Data**  
   Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, Tat Seng Chua. *ICAIF 2024.* [[Paper](https://doi.org/10.1145/3677052.3698685)]
5. **CABINET: Content Relevance based Noise Reduction for Table Question Answering**  
   Sohan Patnaik, Heril Changwal, Milan Aggarwal, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.01155)]
6. **Multimodal Table Understanding**  
   Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, Weiping Wang. *ACL 2024*. [[Paper](https://aclanthology.org/2024.acl-long.493/)]
7. **TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy**  
   Weichao Zhao, Hao Feng, Qi Liu, et al. *NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2406.01326)]
8. **TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning**  
   Yilun Zhao, Lyuhao Chen, Arman Cohan, Chen Zhao. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.692/)]
9. **ReAcTable: Enhancing ReAct for Table Question Answering**  
   Yunjia Zhang, Jordan Henkel, Avrilia Floratou, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[Paper](https://doi.org/10.14778/3659437.3659452)]
10. **Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**  
    Zilong Wang, Hao Zhang, Chun-Liang Li, et al. *ICLR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2401.04398)]
11. **Table-GPT: Table Fine-tuned GPT for Diverse Table Tasks**  
    Peng Li, Yeye He, Dror Yashar, et al. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024*. [[Paper](https://dl.acm.org/doi/10.1145/3654979)]
12. **TableGPT2: A Large Multimodal Model with Tabular Data Integration**  
    Aofeng Su, Aowen Wang, Chao Ye, Chen Zhou, et al. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2411.02059)]
13. **S3HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering**   
    Fangyu Lei, Xiang Li, Yifan Wei, Shizhu He, Yiming Huang, Jun Zhao, Kang Liu. *ACL 2023.* [[Paper](https://aclanthology.org/2023.acl-short.147/)]
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
   Yuanyuan Liang, Tingyu Xie, Gan Peng, Zihao Huang, Yunshi Lan, Weining Qian. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.10434)]
3. **Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks**  
   Xunkai Li, Zhengyu Wu, Jiayi Wu, Hanwen Cui, Jishuo Jia, Rong-Hua Li, Guoren Wang. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2412.12456)]
4. **Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey**  
   Qizhi Pei, Lijun Wu, Kaiyuan Gao, Jinhua Zhu, Yue Wang, Zun Wang, Tao Qin, Rui Yan. *arXiv 2024*. [[Paper](https://arxiv.org/abs/2403.01528)]

###### LLM-based Semantic Analysis

1. **GraphGPT: Graph Instruction Tuning for Large Language Models**   
   Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, Chao Huang. *SIGIR 2024.* [[Paper](https://doi.org/10.48550/arXiv.2310.13023)]
2. **Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models**   
   Guanming Xiong, Junwei Bao, Wen Zhao. *ACL 2024.* [[Paper](https://aclanthology.org/2024.acl-long.569/)]
3. **FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering**     
   Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang. *AAAI 2024.* [[Paper](https://doi.org/10.48550/arXiv.2308.12060)]
4. **Language is All a Graph Needs**   
   Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang. *EACL 2024.* [[Paper](https://aclanthology.org/2024.findings-eacl.132/)]
5. **InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment**   
   Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, Julian McAuley. *Findings of ACL 2024.* [[Paper](https://aclanthology.org/2024.findings-acl.801/)]
6. **Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments**  
   Sitao Cheng, Ziyuan Zhuang, Yong Xu, et a;. *Findings of ACL 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.08593)]
7. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**   
   Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. *NeurIPS 2023.* [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)]
8. **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**   
   Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen. *EMNLP 2023.* [[Paper](https://doi.org/10.48550/arXiv.2305.09645)]
9. **UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph**   
   Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen. *ICLR 2023.* [[Paper](https://doi.org/10.48550/arXiv.2212.00959)]
10. **Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering**   
    Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, Hong Chen. *ACL 2022.* [[Paper](https://aclanthology.org/2022.acl-long.396/)]
11. **RoBERTa: A Robustly Optimized BERT Pretraining Approach**  
    Yinhan Liu, Myle Ott, Naman Goyal, et al. *arXiv 2019*. [[Paper](https://arxiv.org/abs/1907.11692)]
12. **Inductive representation learning on large graphs**   
    William L. Hamilton, Rex Ying, Jure Leskovec. *NeurIPS 2017.* [[Paper](https://dl.acm.org/doi/10.5555/3294771.3294869)]
13. **Semi-Supervised Classification with Graph Convolutional Networks**   
    Thomas N. Kipf, Max Welling. *ICLR 2017.* [[Paper](https://doi.org/10.48550/arXiv.1609.02907)]



#### 4.2.2 LLM for Semi-Structured Data Analysis

1. **Querying Semi-Structured Data**  
   Serge Abiteboul. *ICDT 1997.* [[Paper](https://dl.acm.org/doi/10.5555/645502.656103)]

##### 4.2.2.1 Markup Language

##### 4.2.2.2 Semi-Structured Tables

1. **MiMoTable: A Multi-scale Spreadsheet Benchmark with Meta Operations for Table Reasoning**  
   Zheng Li, Yang Du, Mao Zheng, Mingyang Song. *COLING 2025.* [[Paper](https://doi.org/10.48550/arXiv.2412.11711)]
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
 Y. Lin, M. Hulsebos, R. Ma, S. Shankar, S. Zeighami, A. G. Parameswaran, E. Wu. *arxiv 2024.* [[Paper](https://arxiv.org/abs/2405.04674/)]
4. **DocFormerv2: Local Features for Document Understanding**  
   Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, R. Manmatha. *AAAI 2024.* [[Paper](https://doi.org/10.1609/aaai.v38i2.27828)]
5. **mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding**  
   Anwen Hu, Haiyang Xu, Jiabo Ye, et al. *Findings of EMNLP 2024.* [[Paper](https://aclanthology.org/2024.findings-emnlp.175/)]
6. **DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding**  
   Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang Zhou, Houqiang Li, Can Huang. *SCIS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2311.11810)]
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
   Zhongxin Liu, Zhijie Tang, Junwei Zhang, Xin Xia, Xiaohu Yang. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3639142)]
2. **Large Language Model for Vulnerability Detection: Emerging Results and Future Directions**  
   Xin Zhou, Ting Zhang, David Lo. *ICSE-NIER 2024.* [[Paper](https://doi.org/10.1145/3639476.3639762)]
3. **Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code**  
   Junwei Zhang, Zhongxin Liu, Xing Hu, Xin Xia, Shanping Li. *IEEE TSE 2023.* [[Paper](https://ieeexplore.ieee.org/document/10153647)]
4. **Software Vulnerability Detection with GPT and In-Context Learning**  
   Zhihong Liu, Qing Liao, Wenchao Gu, Cuiyun Gao. *DSC 2023.* [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
5. **CodeBERT: A Pre-Trained Model for Programming and Natural Languages**  
   Zhangyin Feng, Daya Guo, Duyu Tang, et al. *Findings of EMNLP 2020.* [[Paper](https://aclanthology.org/2020.findings-emnlp.139/)]
6. **The Probabilistic Relevance Framework: BM25 and Beyond**  
   Stephen Robertson, Hugo Zaragoza. *Foundations and Trends in Information Retrieval, Volume 3, Issue 4, 2009.* [[Paper](https://dl.acm.org/doi/10.1561/1500000019)]

###### LLM-based Semantic-aware Analysis

1. **Improving Code Summarization With Tree Transformer Enhanced by Position-Related Syntax Complement**  
   Jie Song, Zexin Zhang, Zirui Tang, Shi Feng, Yu Gu. *IEEE TAI 2024.* [[Paper](https://ieeexplore.ieee.org/document/10510878/metrics#metrics)]
2. **Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning**  
   Mingyang Geng, Shangwen Wang, Dezun Dong, et al. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3608134)]
3. **Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)**  
   Toufique Ahmed, Kunal Suresh Pai, Premkumar Devanbu, Earl Barr. *ICSE 2024.* [[Paper](https://doi.org/10.1145/3597503.3639183)]
4. **CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context**  
   Yangruibo Ding, Zijian Wang, Wasi Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang. *LREC-COLING 2024.* [[Paper](https://aclanthology.org/2024.lrec-main.305/)]
5. **Repoformer: Selective Retrieval for Repository-Level Code Completion**  
   Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, Xiaofei Ma. *ICML 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.10059)]
6. **SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation**  
   Yingjie Mao, Xiaoqi Li, Wenkai Li, Xin Wang, Lei Xie. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2402.04863)]
7. **Code Structure–Guided Transformer for Source Code Summarization**  
   Shuzheng Gao, Cuiyun Gao, Yulan He, Jichuan Zeng, Lunyiu Nie, Xin Xia, Michael Lyu. *ACM Transactions on Software Engineering and Methodology 2023.* [[Paper](https://doi.org/10.1145/3522674)]
8. **RepoFusion: Training Code Models to Understand Your Repository**  
   Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, Torsten Scholak. *arXiv 2023.* [[Paper](https://doi.org/10.48550/arXiv.2306.10998)]



### 4.3 LLM for Data System Optimization

[⬆️top](#table-of-contents)

#### 4.3.1 LLM for Configuration Tuning

1. **ELMo-Tune-V2: LLM-Assisted Full-Cycle Auto-Tuning to Optimize LSM-Based Key-Value Stores**  
 Viraj Thakkar, Qi Lin, Kenanya Keandra Adriel Prasetyo, et al. *arxiv 2025* [[Paper](https://arxiv.org/abs/2502.17606)]
2. **Breaking It Down: An In-Depth Study of Index Advisors**  
   Wei Zhou, Chen Lin, Xuanhe Zhou, Guoliang Li. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[Paper](https://dl.acm.org/doi/10.14778/3675034.3675035)]
3. **TRAP: Tailored Robustness Assessment for Index Advisors via Adversarial Perturbation**  
   Wei Zhou; Chen Lin; Xuanhe Zhou; Guoliang Li; Tianqing Wang. *2024 IEEE 40th International Conference on Data Engineering (ICDE)*. [[Paper](https://ieeexplore.ieee.org/document/10597867)]
4. **Automatic Database Knob Tuning: A Survey**  
   Xinyang Zhao, Xuanhe Zhou, Guoliang Li. *IEEE Transactions on Knowledge and Data Engineering, Volume 35, Issue 12. 2023.*  [[Paper](https://dl.acm.org/doi/10.1109/TKDE.2023.3266893)]
5. **Demonstration of ViTA: Visualizing, Testing and Analyzing Index Advisors**    
   Wei Zhou, Chen Lin, Xuanhe Zhou, Guoliang Li, Tianqing Wang. *CIKM 2023.* [[Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614738)]
6. **An Efficient Transfer Learning Based Configuration Adviser for Database Tuning**  
   Xinyi Zhang, Hong Wu, Yang Li, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 3. 2023.* [[Paper](https://dl.acm.org/doi/abs/10.14778/3632093.3632114)]
7. **Code-aware cross-program transfer hyperparameter optimization**  
   Zijia Wang, Xiangyu He, Kehan Chen, Chen Lin, Jinsong Su. *AAAI 2023.* [[Paper](https://dl.acm.org/doi/10.1609/aaai.v37i9.26226)]
8. **QTune: a query-aware database tuning system with deep reinforcement learning**  
   Guoliang Li, Xuanhe Zhou, Shifu Li, Bo Gao. *Proceedings of the VLDB Endowment, Volume 12, Issue 12. 2019.* [[Paper](https://dl.acm.org/doi/10.14778/3352063.3352129)]

##### Tuning Task-Aware Prompt Engineering

1. **λ-Tune: Harnessing Large Language Models for Automated Database System Tuning**  
   Victor Giannankouris, Immanuel Trummer. *SIGMOD 2025.* [[Paper](https://doi.org/10.48550/arXiv.2411.03500)]
2. **LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model**  
   Xinxin Zhao, Haoyang Li, Jing Zhang, et al. *arxiv 2025.* [[Paper](https://arxiv.org/abs/2503.07884)]
3. **LATuner: An LLM-Enhanced Database Tuning System Based on Adaptive Surrogate Model**  
   Chongjiong Fan, Zhicheng Pan, Wenwen Sun, Chengcheng Yang, Wei-Neng Chen. *ECML PKDD 2024.* [[Paper](https://doi.org/10.1007/978-3-031-70362-1_22)]
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



#### 4.3.2 LLM for Query Optimization

##### Optimization-Aware Prompt Engineering

1. **QUITE: A Query Rewrite System Beyond Rules with LLM Agents**  
   Yuyang Song, Hanxu Yan, Jiale Lao, Yibo Wang, et al. *arXiv 2025.* [[Paper](https://arxiv.org/pdf/2506.07675)]
2. **Can Large Language Models Be Query Optimizer for Relational Databases?**  
   Jie Tan, Kangfei Zhao, Rui Li, et al. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.05562)]
3. **A Query Optimization Method Utilizing Large Language Models**  
   Zhiming Yao, Haoyang Li, Jing Zhang, Cuiping Li, Hong Chen. *arxiv 2025.* [[Paper](https://arxiv.org/abs/2503.06902)]
4. **Query Rewriting via LLMs**  
   Sriram Dharwada, Himanshu Devrani, Jayant Haritsa, Harish Doraiswamy. *arXiv 2025.* [[Paper](https://doi.org/10.48550/arXiv.2502.12918)]
5. **DB-GPT: Large Language Model Meets Database** [[Paper](https://link.springer.com/article/10.1007/s41019-023-00235-6)]
6. **LLM-R2: A Large Language Model Enhanced Rule-Based Rewrite System for Boosting Query Efficiency**  
   Zhaodonghui Li, Haitao Yuan, Huiming Wang, Gao Cong, Lidong Bing. *VLDB 2024.* [[Paper](https://doi.org/10.14778/3696435.3696440)]
7. **The Unreasonable Effectiveness of LLMs for Query Optimization**  
   Peter Akioyamen, Zixuan Yi, Ryan Marcus. *ML for Systems Workshop at NeurIPS 2024.* [[Paper](https://doi.org/10.48550/arXiv.2411.02862)]
8. **R-Bot: An LLM-based Query Rewrite System**  
   Zhaoyan Sun, Xuanhe Zhou, Guoliang Li. *arXiv 2024.* [[Paper](https://arxiv.org/abs/2412.01661)]
9. **Query Rewriting via Large Language Models**  
   Jie Liu, Barzan Mozafari. *arXiv 2024.* [[Paper](https://doi.org/10.48550/arXiv.2403.09060)]



#### 4.3.3 LLM for Anomaly Diagnosis

##### Manually Crafted Prompts for Anomaly Diagnosis

1. **DBG-PT: A Large Language Model Assisted Query Performance Regression Debugger**  
   Victor Giannakouris, Immanuel Trummer. *Proceedings of the VLDB Endowment, Volume 17, Issue 12, 2024.* [[Paper](https://doi.org/10.14778/3685800.3685869)]

##### RAG Based Diagnosis Experience Enrichment

1. **Query Performance Explanation through Large Language Model for HTAP Systems**   
   Haibo Xiu, Li Zhang, Tieying Zhang, Jun Yang, Jianjun Chen. *ICDE 2025.* [[Paper](https://doi.org/10.48550/arXiv.2412.01709)]
2. **D-Bot: Database Diagnosis System using Large Language Models**  
   Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, et al. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
3. **LLM As DBA**  
   Xuanhe Zhou, Guoliang Li, Zhiyuan Liu. *arXiv 2023.* [[Paper](https://arxiv.org/abs/2308.05481)]

##### Multi-Agent Mechanism for Collaborative Diagnosis

1. **D-Bot: Database Diagnosis System using Large Language Models** [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
2. **Panda: Performance Debugging for Databases using LLM Agents**  
   Vikramank Singh, Kapil Eknath Vaidya, Vinayshekhar Bannihatti Kumar, et al. *CIDR 2024.* [[Paper](https://www.cidrdb.org/cidr2024/papers/p6-singh.pdf)]
3. **LLM As DBA** [[Paper](https://arxiv.org/abs/2308.05481)]

##### Localized LLM Enhancement via Specialized FineTuning

1. **D-Bot: Database Diagnosis System using Large Language Models** [[Paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
