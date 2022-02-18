# MLC Paper List

A curated list of papers multi-label classification papers.

~~strikethrough~~ means doubtful results or methods.

Paperswithcode links will be preferred.

## Related Resources

- [AAAI 2022 Accepted Papers](https://aaai.org/Conferences/AAAI-22/wp-content/uploads/2021/12/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf)
- Semantic Scholar: https://www.semanticscholar.org
- Google ScholarL: https://scholar.google.com
- Bing Academic: https://cn.bing.com/academic
- Connected Papers: https://www.connectedpapers.com
- Papers with Code: https://paperswithcode.com
- Research Papers: https://papers.labml.ai/conferences
- Deeplearning Monitor: https://deeplearn.org
- OpenReview: https://openreview.net

## Datasets

| Cat. | Name          | Title                                                                          |                                     Links                                      |
|:----:| ------------- | ------------------------------------------------------------------------------ |:------------------------------------------------------------------------------:|
| NLP  | RCV1-v2       | Rcv1: A new benchmark collection for text categorization research              |        [[paper]](https://jmlr.org/papers/volume5/lewis04a/lewis04a.pdf)        |
| NLP  | AAPD          | SGM: Sequence Generation Model for Multi-label Classification                  |                                                                                |
| NLP  | Reuters-21578 | An analysis of the relative hardness of Reuters‐21578 subsets                  |                                                                                |
| NLP  | RMSC          | Driven Multi-Label Music Style Classification by Exploiting Style Correlations |                                                                                |
| NLP  | Emotion       | GoEmotions: A dataset of fine-grained emotions                                 |                                                                                |
|  CV  | Sewer-ML      | Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark      | [[link]](https://paperswithcode.com/paper/sewer-ml-a-multi-label-sewer-defect) |
|  CV  | MS-COCO       | Microsoft COCO: Common Objects in Context                                      |                                                                                |

## Transformer-Based

| Cat. |     Pub.     | Year | Title                                                                                           |                                                 Links                                                 |
|:----:|:------------:|:----:| ----------------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------------------:|
| NLP  |   ACL ARR    | 2022 | HCL-MTC Hierarchical Contrastive Learning for Multi-label Text Classification                   |                         [[paper]](https://openreview.net/pdf?id=R1BifFIieBP)                          |
| NLP  |    Arxiv     | 2021 | Hierarchy Decoder is All You Need To Text Classification                                        |                           [[paper]](https://arxiv.org/pdf/2111.11104v1.pdf)                           |
|  CV  |    Arxiv     | 2021 | ML-Decoder: Scalable and Versatile Classification Head                                          | [[paper]](https://arxiv.org/pdf/2111.12933v1.pdf)[[code]](https://github.com/alibaba-miil/ml_decoder) |
|  CV  |    Arxiv     | 2021 | Query2Label: A Simple Transformer Way to Multi-Label Classification                             |  [[paper]](https://arxiv.org/pdf/2107.10834v1.pdf)[[code]](https://github.com/SlongLiu/query2labels)  |
|  CV  |    Arxiv     | 2021 | MlTr: Multi-label Classification with Transformer                                               |     [[paper]](https://arxiv.org/pdf/2106.06195v1.pdf)[[code]](https://github.com/starmemda/MlTr)      |
| NLP  |    Arxiv     | 2021 | Hierarchy-Aware T5 with Path-Adaptive Mask Mechanism for Hierarchical Text Classification       |                            [[paper]](https://arxiv.org/pdf/2109.08585.pdf)                            |
| NLP  |    Arxiv     | 2021 | Label Mask for Multi-Label Text Classification                                                  |                            [[paper]](https://arxiv.org/pdf/2106.10076.pdf)                            |
|  CV  |     ICCV     | 2021 | Transformer-based Dual Relation Graph for Multi-label Image Recognition                         |        [[link]](https://paperswithcode.com/paper/transformer-based-dual-relation-graph-for-1)         |
| NLP  |    IJCAI     | 2021 | Correlation-Guided Representation for Multi-Label Text Classification                           |                      [[paper]](https://www.ijcai.org/proceedings/2021/0463.pdf)                       |
| NLP  | ACL Findings | 2021 | Enhancing Label Correlation Feedback in Multi-Label Text Classification via Multi-Task Learning |      [[paper]](https://arxiv.org/pdf/2106.03103.pdf)[[code]](https://github.com/EiraZhang/LACO)       |
|  CV  |     CVPR     | 2021 | General Multi-Label Image Classification With Transformers                                      |        [[link]](https://paperswithcode.com/paper/general-multi-label-image-classification-with)         |
|  CV  |     ACMM     | 2021 | M3TR: Multi-modal Multi-label Recognition with Transformer                                      |                     [[paper]](https://dl.acm.org/doi/pdf/10.1145/3474085.3475191)                     |

## Graph-Based

| Cat. |  Pub.   | Year | Title                                                                                                                            |                                                  Links                                                  |
| ---- |:-------:|:----:| -------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------:|
| NLP  | ACL ARR | 2021 | Multi-Label Text Classification by Graph Neural Network with Mixing Operations                                                   |                           [[paper]](https://openreview.net/pdf?id=XT4BaluDTo)                           |
| NLP  |  Arxiv  | 2021 | Heterogeneous Graph Neural Networks for Multi-label Text Classification                                                          |                            [[paper]](https://arxiv.org/pdf/2103.14620v1.pdf)                            |
| NLP  |   ACL   | 2021 | Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification                                            |                        [[paper]](https://aclanthology.org/2021.acl-long.337.pdf)                        |
| NLP  |   ACL   | 2021 | Label-Specific Dual Graph Neural Network for Multi-Label Text Classification                                                     |                        [[paper]](https://aclanthology.org/2021.acl-long.298.pdf)                        |
| NLP  |  EMNLP  | 2021 | Beyond Text: Incorporating Metadata and Label Structure for Multi-Label Document Classification using Heterogeneous Graphs       |                       [[paper]](https://aclanthology.org/2021.emnlp-main.253.pdf)                       |
| NLP  |  EMNLP  | 2021 | Hierarchical Multi-label Text Classification with Horizontal and Vertical Category Correlations                                  |                       [[paper]](https://aclanthology.org/2021.emnlp-main.190.pdf)                       |
| NLP  |  Arxiv  | 2021 | Heterogeneous Graph Neural Networks for Multi-label Text Classification                                                          |                            [[paper]](https://arxiv.org/pdf/2103.14620v1.pdf)                            |
| NLP  |  WSDM   | 2020 | Beyond Statistical Relations: Integrating Knowledge Relations into Style Correlations for Multi-Label Music Style Classification |   [[paper]](https://arxiv.org/pdf/1911.03626v2.pdf)[[code]](https://github.com/Makwen1995/MusicGenre)   |
| NLP  |   ACL   | 2019 | Hierarchy-Aware Global Model for Hierarchical Text Classification                                                                | [[paper]](https://aclanthology.org/2020.acl-main.104.pdf)[[code]](https://github.com/Alibaba-NLP/HiAGM) |
| CV   |  CVPR   | 2019 | Multi-Label Image Recognition with Graph Convolutional Networks                                                                  |           [[link]](https://paperswithcode.com/paper/multi-label-image-recognition-with-graph)           |
| CV   |  CVPR   | 2018 | Multi-Label Zero-Shot Learning with Structured Knowledge Graphs                                                                  |             [[link]](https://paperswithcode.com/paper/multi-label-zero-shot-learning-with)              |

## Loss-Based

| Cat. | Pub.  | Year | Title                                                                                     |                                                     Links                                                     |
|:----:|:-----:|:----:| ----------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------:|
|  CV  | Arxiv | 2021 | Multi-label Classification with Partial Annotations using Class-aware Selective Loss      | [[paper]](https://arxiv.org/pdf/2110.10955v1.pdf)[[code]](https://github.com/alibaba-miil/partiallabelingcsl) |
| NLP  | EMNLP | 2021 | Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution | [[paper]](https://aclanthology.org/2021.emnlp-main.643.pdf)[[code]](https://github.com/Roche/BalancedLossNLP) |
|  CV  | ICCV  | 2021 | Asymmetric Loss For Multi-Label Classification                                            |          [[paer]](https://arxiv.org/pdf/2009.14119.pdf)[[code]](https://github.com/Alibaba-MIIL/ASL)          |

## XMLC

| Cat. | Pub. | Year | Title                                                                                                             |                                            Links                                             |
|:----:|:----:|:----:| ----------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------:|
| NLP  | AAAI | 2021 | LightXML: Transformer with Dynamic Negative Sampling for High-Performance Extreme Multi-label Text Classification |  [[paper]](https://arxiv.org/pdf/2101.03305.pdf)[[code]](http://github.com/kongds/LightXML)  |
| NLP  | KDD  | 2020 | Taming Pretrained Transformers for Extreme Multi-label Text Classification                                        | [[paper]](https://arxiv.org/pdf/1905.02331.pdf)[[code]](https://github.com/guoqunabc/X-BERT) |
| NLP  | KDD  | 2020 | Correlation Networks for Extreme Multi-label Text Classification                                                  |                [[paper]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403151)                 |

## Partial Label (promising)

| Cat. | Pub. | Year | Title                                                                                        |                                                Links                                                 |
|:----:|:----:|:----:| -------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------:|
|  CV  | AAAI | 2022 | Structured Semantic Transfer for Multi-Label Recognition with Partial Labels                 | [[paper]](https://arxiv.org/pdf/2112.10941v2.pdf)[[code]](https://github.com/hcplab-sysu/hcp-mlr-pl) |
|  CV  | AAAI | 2022 | Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels |                                                                                                      |

## Other

| Cat. | Pub.  | Year | Title                                                                                                              |                                                              Links                                                              |
| ---- | ----- | ---- | ------------------------------------------------------------------------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------:|
| NLP  | IPM   | 2021 | A novel reasoning mechanism for multi-label text classification                                                    |                         [[paper]](https://www.sciencedirect.com/science/article/pii/S0306457320309341)                          |
| CV   | Arxiv | 2021 | Multi-layered Semantic Representation Network for Multi-label Image Classification                                 |                  [[paper]](https://arxiv.org/pdf/2106.11596v1.pdf)[[code]](https://github.com/chehao2628/MSRN)                  |
| NLP  | EACL  | 2021 | Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification                          | [[paper]](https://aclanthology.org/2021.eacl-main.247.pdf)[[code]](https://github.com/soumyac1999/hyperbolic-label-emb-for-hmc) |
| NLP  | NAACL | 2021 | Modeling Diagnostic Label Correlation for Automatic ICD Coding                                                     |         [[paper]](https://aclanthology.org/2021.naacl-main.318.pdf)[[code]](https://github.com/MiuLab/ICD-Correlation)          |
| NLP  | NAACL | 2021 | HTCInfoMax: A Global Model for Hierarchical Text Classification via Information Maximization                       |         [[paper]](https://aclanthology.org/2021.naacl-main.260.pdf)[[code]](https://github.com/RingBDStack/HTCInfoMax)          |
| CV   | ICCV  | 2021 | Residual Attention: A Simple but Effective Method for Multi-Label Recognition                                      |                      [[link]](https://paperswithcode.com/paper/residual-attention-a-simple-but-effective)                       |
| NLP  | NLPCC | 2020 | Label-Wise Document Pre-Training for Multi-Label Text Classification                                               |                  [[paper]](https://arxiv.org/pdf/2008.06695v1.pdf)[[code]](https://github.com/laddie132/LW-PT)                  |
| CV   | AAAI  | 2020 | Multi-Label Classification with Label Graph Superimposing                                                          |                  [[paper]](https://arxiv.org/pdf/1911.09243v1.pdf)[[code]](https://github.com/mathkey/mssnet)                   |
| NLP  | NAACL | 2016 | Improved Neural Network-based Multi-label Classification with Better Initialization Leveraging Label Co-occurrence |                                        [[paper]](https://aclanthology.org/N16-1063.pdf)                                         |