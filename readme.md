# MLC Paper List

A curated list of multi-label classification papers.

Paperswithcode links will be preferred.

Welcome contributions!

## Datasets

| Cat. |     Name      | Title                                                                          |                                          Links                                           |
|:----:|:-------------:| ------------------------------------------------------------------------------ |:----------------------------------------------------------------------------------------:|
| NLP  |    RCV1-v2    | Rcv1: A new benchmark collection for text categorization research              |                    [[link]](https://paperswithcode.com/dataset/rcv1)                     |
| NLP  |     AAPD      | SGM: Sequence Generation Model for Multi-label Classification                  | [[link]](https://paperswithcode.com/paper/sgm-sequence-generation-model-for-multi-label) |
| NLP  |      WOS      | HDLTex: Hierarchical Deep Learning for Text Classification                     |  [[link]](https://paperswithcode.com/paper/hdltex-hierarchical-deep-learning-for-text)   |
| NLP  | Reuters-21578 | An analysis of the relative hardness of Reuters‐21578 subsets                  |                [[link]](https://paperswithcode.com/dataset/reuters-21578)                |
| NLP  |     RMSC      | Driven Multi-Label Music Style Classification by Exploiting Style Correlations |     [[link]](https://paperswithcode.com/paper/review-driven-multi-label-music-style)     |
| NLP  |  GoEmotions   | GoEmotions: A dataset of fine-grained emotions                                 | [[link]](https://paperswithcode.com/paper/goemotions-a-dataset-of-fine-grained-emotions) |
|  CV  |   Sewer-ML    | Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark      |      [[link]](https://paperswithcode.com/paper/sewer-ml-a-multi-label-sewer-defect)      |
|  CV  |    MS-COCO    | Microsoft COCO: Common Objects in Context                                      |   [[link]](https://paperswithcode.com/paper/microsoft-coco-common-objects-in-context)    |
|  MM  |     M3ED      | M3ED: Multi-modal Multi-scene Multi-label Emotional Dialogue Database          |   [[link]](https://paperswithcode.com/paper/m3ed-multi-modal-multi-scene-multi-label)    |

## Survey

| Year | Title                                                          |                      Links                       |
| ---- | -------------------------------------------------------------- |:------------------------------------------------:|
| 2020 | The Emerging Trends of Multi-Label Learning                    | [[paper]](https://arxiv.org/pdf/2011.11197.pdf)  |
| 2020 | A Survey on Text Classification: From Shallow to Deep Learning | [[link]](https://arxiv.org/pdf/2008.00364v6.pdf) |
| 2019 | Survey on Multi-Output Learning                                |    [[paper]](http://arxiv.org/pdf/1901.00248)    |

## Toolkit

| Year | Title                                                                                        |                                                      Links                                                      |
| ---- | -------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------:|
| 2021 | PaddleNLP: An Easy-to-use and High Performance NLP Library | [[repo]](https://github.com/PaddlePaddle/PaddleNLP)[[code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label)
| 2019 | NeuralClassifier: An Open-source Neural Hierarchical Multi-label Text Classification Toolkit | [[paper]](https://aclanthology.org/P19-3015.pdf)[[code]](https://github.com/Tencent/NeuralNLP-NeuralClassifier) |

## Models

### Prompt-Based

| Cat. |     Pub.     | Year | Title                                                                                                                      |                                                                                        Links                                                                                        |
|:----:|:------------:|:----:| -------------------------------------------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| NLP  |    Arxiv     | 2022 | HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification                                                    |                                                                  [[paper]](https://arxiv.org/pdf/2204.13413v1.pdf)                                                                  |
| NLP  |    Springer     | 2021 | Label prompt for multi-label text classification                                                                             |                                                                   [[paper]](https://link.springer.com/article/10.1007/s10489-022-03896-4)                                                                   |

>PaddleNLP Solution: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label/few-shot

### Transformer-Based

| Cat. |     Pub.     | Year | Title                                                                                                                      |                                                                                        Links                                                                                        |
|:----:|:------------:|:----:| -------------------------------------------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| NLP  |     ACL      | 2022 | Improving Multi-label Malevolence Detection in Dialogues through Multi-faceted Label Correlation Enhancement               |                                               [[link]](https://paperswithcode.com/paper/improving-multi-label-malevolence-detection)                                                |
| NLP  |     ACL      | 2022 | Contrastive Learning-Enhanced Nearest Neighbor Mechanism for Multi-Label Text Classification                               |                                                  [[link]](https://paperswithcode.com/paper/contrastive-learning-enhanced-nearest)                                                   |
|  MM  |     AAAI     | 2022 | Tailor Versatile Multi-modal Learning for Multi-label Emotion Recognition                                                  |                                             [[paper]](https://arxiv.org/pdf/2201.05834.pdf)[[code]](https://github.com/kniter1/TAILOR)                                              |
| NLP  |    ICASSP    | 2022 | Multi-relation Message Passing for Multi-label Text Classification                                                         |                                                 [[link]](https://paperswithcode.com/paper/multi-relation-message-passing-for-multi)                                                 |
| NLP  |     ECIR     | 2021 | A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers                          | [[paper]](https://annefried.github.io/files/hierarchical_patent_classification_ecir2021.pdf)[[code]](https://github.com/boschresearch/hierarchical_patent_classification_ecir2021/) |
|  CV  |     ICPR     | 2021 | Visual Transformers with Primal Object Queries for Multi-Label Image Classification                                        |                                                  [[link]](https://paperswithcode.com/paper/visual-transformers-with-primal-object)                                                  |
| NLP  |    Arxiv     | 2021 | Hierarchy Decoder is All You Need To Text Classification                                                                   |                                                                  [[paper]](https://arxiv.org/pdf/2111.11104v1.pdf)                                                                  |
|  CV  |    Arxiv     | 2021 | ML-Decoder: Scalable and Versatile Classification Head                                                                     |                                                    [[link]](https://paperswithcode.com/paper/ml-decoder-scalable-and-versatile)                                                     |
|  CV  |    Arxiv     | 2021 | Query2Label: A Simple Transformer Way to Multi-Label Classification                                                        |                                              [[link]](https://paperswithcode.com/paper/query2label-a-simple-transformer-way-to-multi)                                               |
|  CV  |    Arxiv     | 2021 | MlTr: Multi-label Classification with Transformer                                                                          |                                                   [[link]](https://paperswithcode.com/paper/mltr-multi-label-classification-with)                                                   |
| NLP  |     ICDM     | 2021 | Expert Knowledge-Guided Length-Variant Hierarchical Label Generation for Proposal Classification                           |                                                                  [[paper]](https://arxiv.org/pdf/2109.06661v2.pdf)                                                                  |
| NLP  |    Arxiv     | 2021 | Hierarchy-Aware T5 with Path-Adaptive Mask Mechanism for Hierarchical Text Classification                                  |                                                                   [[paper]](https://arxiv.org/pdf/2109.08585.pdf)                                                                   |
|  CV  |     ICCV     | 2021 | Transformer-based Dual Relation Graph for Multi-label Image Recognition                                                    |                                               [[link]](https://paperswithcode.com/paper/transformer-based-dual-relation-graph-for-1)                                                |
| NLP  |    IJCAI     | 2021 | Correlation-Guided Representation for Multi-Label Text Classification                                                      |                                                             [[paper]](https://www.ijcai.org/proceedings/2021/0463.pdf)                                                              |
| NLP  | ACL Findings | 2021 | Enhancing Label Correlation Feedback in Multi-Label Text Classification via Multi-Task Learning                            |                                              [[link]](https://paperswithcode.com/paper/enhancing-label-correlation-feedback-in-multi)                                               |
|  CV  |     CVPR     | 2021 | General Multi-Label Image Classification With Transformers                                                                 |                                              [[link]](https://paperswithcode.com/paper/general-multi-label-image-classification-with)                                               |
|  CV  |     AAAI     | 2021 | HOT-VAE: Learning High-Order Label Correlation for Multi-Label Classification via Attention-Based Variational Autoencoders |                                              [[link]](https://paperswithcode.com/paper/hot-vae-learning-high-order-label-correlation)                                               |
|  MM  |    ACM MM    | 2021 | M3TR: Multi-modal Multi-label Recognition with Transformer                                                                 |                                                            [[paper]](https://dl.acm.org/doi/pdf/10.1145/3474085.3475191)                                                            |

### Graph-Based

| Cat. |  Pub.   | Year | Title                                                                                                                            |                                                  Links                                                   |
|:----:|:-------:|:----:| -------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------:|
| NLP  |   ACL   | 2022 | Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification                  |          [[link]](https://paperswithcode.com/paper/incorporating-hierarchy-into-text-encoder-a)          |
|  CV  |  AAAI   | 2021 | Modular Graph Transformer Networks for Multi-Label Image Classification                                                          |                    [[paper]](https://people.cs.umu.se/sonvx/files/2021_AAAI_MGTN.pdf)                    |
| NLP  | ACL-ARR | 2021 | Multi-Label Text Classification by Graph Neural Network with Mixing Operations                                                   |                           [[paper]](https://openreview.net/pdf?id=XT4BaluDTo)                            |
| NLP  |  Arxiv  | 2021 | Heterogeneous Graph Neural Networks for Multi-label Text Classification                                                          |                            [[paper]](https://arxiv.org/pdf/2103.14620v1.pdf)                             |
| NLP  |   ACL   | 2021 | Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification                                            | [[paper]](https://aclanthology.org/2021.acl-long.337.pdf)[[code]](https://github.com/RuiBai1999/HiMatch) |
| NLP  |   ACL   | 2021 | Label-Specific Dual Graph Neural Network for Multi-Label Text Classification                                                     |                        [[paper]](https://aclanthology.org/2021.acl-long.298.pdf)                         |
| NLP  |  EMNLP  | 2021 | Beyond Text: Incorporating Metadata and Label Structure for Multi-Label Document Classification using Heterogeneous Graphs       |                       [[paper]](https://aclanthology.org/2021.emnlp-main.253.pdf)                        |
| NLP  |  EMNLP  | 2021 | Hierarchical Multi-label Text Classification with Horizontal and Vertical Category Correlations                                  |                       [[paper]](https://aclanthology.org/2021.emnlp-main.190.pdf)                        |
| NLP  |  Arxiv  | 2021 | Heterogeneous Graph Neural Networks for Multi-label Text Classification                                                          |                            [[paper]](https://arxiv.org/pdf/2103.14620v1.pdf)                             |
| NLP  |  WSDM   | 2020 | Beyond Statistical Relations: Integrating Knowledge Relations into Style Correlations for Multi-Label Music Style Classification |   [[paper]](https://arxiv.org/pdf/1911.03626v2.pdf)[[code]](https://github.com/Makwen1995/MusicGenre)    |
| NLP  |   ACL   | 2019 | Hierarchy-Aware Global Model for Hierarchical Text Classification                                                                | [[paper]](https://aclanthology.org/2020.acl-main.104.pdf)[[code]](https://github.com/Alibaba-NLP/HiAGM)  |
|  CV  |  CVPR   | 2019 | Multi-Label Image Recognition with Graph Convolutional Networks                                                                  |           [[link]](https://paperswithcode.com/paper/multi-label-image-recognition-with-graph)            |
|  CV  |  CVPR   | 2018 | Multi-Label Zero-Shot Learning with Structured Knowledge Graphs                                                                  |              [[link]](https://paperswithcode.com/paper/multi-label-zero-shot-learning-with)              |

### Loss-Based

| Cat. | Pub.  | Year | Title                                                                                     |                                                     Links                                                     |
|:----:|:-----:|:----:| ----------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------:|
|  CV  | Arxiv | 2021 | Multi-label Classification with Partial Annotations using Class-aware Selective Loss      | [[paper]](https://arxiv.org/pdf/2110.10955v1.pdf)[[code]](https://github.com/alibaba-miil/partiallabelingcsl) |
| NLP  | EMNLP | 2021 | Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution | [[paper]](https://aclanthology.org/2021.emnlp-main.643.pdf)[[code]](https://github.com/Roche/BalancedLossNLP) |
|  CV  | ICCV  | 2021 | Asymmetric Loss For Multi-Label Classification                                            |          [[paper]](https://arxiv.org/pdf/2009.14119.pdf)[[code]](https://github.com/Alibaba-MIIL/ASL)          |

### XMLC

| Cat. | Pub. | Year | Title                                                                                                             |                                            Links                                             |
|:----:|:----:|:----:| ----------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------:|
| NLP  | ACL  | 2022 | Evaluating Extreme Hierarchical Multi-label Classification                                                        |    [[link]](https://paperswithcode.com/paper/evaluating-extreme-hierarchical-multi-label)    |
| NLP  | AAAI | 2021 | LightXML: Transformer with Dynamic Negative Sampling for High-Performance Extreme Multi-label Text Classification |  [[paper]](https://arxiv.org/pdf/2101.03305.pdf)[[code]](http://github.com/kongds/LightXML)  |
| NLP  | KDD  | 2020 | Taming Pretrained Transformers for Extreme Multi-label Text Classification                                        | [[paper]](https://arxiv.org/pdf/1905.02331.pdf)[[code]](https://github.com/guoqunabc/X-BERT) |
| NLP  | KDD  | 2020 | Correlation Networks for Extreme Multi-label Text Classification                                                  |                [[paper]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403151)                 |

### Partial Label (promising)

| Cat. |   Pub.    | Year | Title                                                                                        |                                                Links                                                 |
|:----:|:---------:|:----:| -------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------:|
|  CV  |   ICLR    | 2022 | Contrastive Label Disambiguation for Partial Label Learning                                  |       [[link]](https://paperswithcode.com/paper/contrastive-label-disambiguation-for-partial)        |
|  CV  |   AAAI    | 2022 | Structured Semantic Transfer for Multi-Label Recognition with Partial Labels                 | [[paper]](https://arxiv.org/pdf/2112.10941v2.pdf)[[code]](https://github.com/hcplab-sysu/hcp-mlr-pl) |
|  CV  |   AAAI    | 2022 | Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels |        [[link]](https://paperswithcode.com/paper/semantic-aware-representation-blending-for)         |
|  CV  | CVPR Workshop | 2021 | PLM: Partial Label Masking for Imbalanced Multi-label Classification                         |                          [[paper]](https://arxiv.org/pdf/2105.10782v1.pdf)                           |


### Few/Zero shot
| Cat. | Pub.  | Year | Title                                                                                                                  |                                                       Links                                                        |
|:----:|:-----:| ---- | ---------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------:|
| NLP  |  KDD  | 2021 | Generalized Zero-Shot Extreme Multi-label Learning                                                                     |               [[link]](https://paperswithcode.com/paper/extreme-zero-shot-learning-for-extreme-text)               |
|  CV  | CVPR  | 2021 | Multi-Label Learning from Single Positive Labels                                                                       | [[paper]](https://arxiv.org/abs/2106.09708.pdf)[[code]](https://github.com/elijahcole/single-positive-multi-label) |
| NLP  | NAACL | 2021 | Improving Pretrained Models for Zero-shot Multi-label Text Classification through Reinforced Label Hierarchy Reasoning |   [[paper]](https://aclanthology.org/2021.naacl-main.83.pdf)[[code]](https://github.com/layneins/Zero-shot-RLHR)   |
| NLP  | EMNLP | 2020 | Multi-label Few/Zero-shot Learning with Knowledge Aggregated from Multiple Label Graphs                                |       [[paper]](https://aclanthology.org/2020.emnlp-main.235.pdf)[[code]](https://github.com/MemoriesJ/KAMG)       |


### Others

| Cat. |  Pub.  | Year | Title                                                                                                              |                                                              Links                                                              |
|:----:|:------:| ---- | ------------------------------------------------------------------------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------:|
| NLP  | NAACL  | 2021 | TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names                                     |                       [[link]](https://paperswithcode.com/paper/taxoclass-hierarchical-multi-label-text)                        |
| NLP  |  ACL   | 2021 | Concept-Based Label Embedding via Dynamic Routing for Hierarchical Text Classification                             |          [[paper]](https://aclanthology.org/2021.acl-long.388.pdf)[[code]](https://github.com/wxpkanon/CLEDforHTC.git)          |
| NLP  |  IPM   | 2021 | A novel reasoning mechanism for multi-label text classification                                                    |                         [[paper]](https://www.sciencedirect.com/science/article/pii/S0306457320309341)                          |
|  CV  | Arxiv  | 2021 | Multi-layered Semantic Representation Network for Multi-label Image Classification                                 |                  [[paper]](https://arxiv.org/pdf/2106.11596v1.pdf)[[code]](https://github.com/chehao2628/MSRN)                  |
| NLP  |  EACL  | 2021 | Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification                          | [[paper]](https://aclanthology.org/2021.eacl-main.247.pdf)[[code]](https://github.com/soumyac1999/hyperbolic-label-emb-for-hmc) |
| NLP  | NAACL  | 2021 | Modeling Diagnostic Label Correlation for Automatic ICD Coding                                                     |         [[paper]](https://aclanthology.org/2021.naacl-main.318.pdf)[[code]](https://github.com/MiuLab/ICD-Correlation)          |
| NLP  | NAACL  | 2021 | HTCInfoMax: A Global Model for Hierarchical Text Classification via Information Maximization                       |         [[paper]](https://aclanthology.org/2021.naacl-main.260.pdf)[[code]](https://github.com/RingBDStack/HTCInfoMax)          |
|  CV  |  ICCV  | 2021 | Residual Attention: A Simple but Effective Method for Multi-Label Recognition                                      |                      [[link]](https://paperswithcode.com/paper/residual-attention-a-simple-but-effective)                       |
| NLP  | NLPCC  | 2020 | Label-Wise Document Pre-Training for Multi-Label Text Classification                                               |                  [[paper]](https://arxiv.org/pdf/2008.06695v1.pdf)[[code]](https://github.com/laddie132/LW-PT)                  |
|  CV  |  AAAI  | 2020 | Multi-Label Classification with Label Graph Superimposing                                                          |                     [[link]](https://paperswithcode.com/paper/multi-label-classification-with-label-graph)                      |
| NLP  | IJCNLP | 2019 | Label-Specific Document Representation for Multi-Label Text Classification                                         |                      [[link]](https://paperswithcode.com/paper/label-specific-document-representation-for)                      |
| NLP  |  ACL   | 2018 | Joint Embedding of Words and Labels for Text Classification                                                        |                     [[link]](https://paperswithcode.com/paper/joint-embedding-of-words-and-labels-for-text)                     |
| NLP  | ICMLA  | 2017 | HDLTex: Hierarchical Deep Learning for Text Classification                                                         |                      [[link]](https://paperswithcode.com/paper/hdltex-hierarchical-deep-learning-for-text)                      |
| NLP  | NAACL  | 2016 | Improved Neural Network-based Multi-label Classification with Better Initialization Leveraging Label Co-occurrence |                                        [[paper]](https://aclanthology.org/N16-1063.pdf)                                         |


## Related Resources

- [AAAI 2022 Accepted Papers](https://aaai.org/Conferences/AAAI-22/wp-content/uploads/2021/12/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf)
- [ACL 2022 Accepted Papers](https://www.2022.aclweb.org/papers)
- [CVPR 2022 Accepted Papers](https://cvpr2022.thecvf.com/accepted-papers)
- Semantic Scholar: https://www.semanticscholar.org
- Google Scholar: https://scholar.google.com
- Bing Academic: https://cn.bing.com/academic
- Connected Papers: https://www.connectedpapers.com
- Papers with Code: https://paperswithcode.com
- Research Papers: https://papers.labml.ai/conferences
- Deeplearning Monitor: https://deeplearn.org
- OpenReview: https://openreview.net
- OpenResearch: https://www.openresearch.org
