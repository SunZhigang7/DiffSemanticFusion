<div align="center">
<h3>DiffSemanticFusion: Semantic Raster BEV Fusion for Autonomous Driving via Online HD Map Diffusion</h3>

Zhigang Sun<sup>1\*</sup>, Yiru Wang<sup>1\*†</sup>, Anqing Jiang<sup>1\*</sup>, Shuo Wang<sup>1</sup>, Yu Gao<sup>1</sup>, Yuwen Heng<sup>1</sup>,  
Shouyi Zhang<sup>1</sup>, An He<sup>1</sup>, Hao Jiang<sup>2</sup>，Jinhao Chai<sup>3</sup>, Zichong Gu<sup>3</sup>, Jijun Wang<sup>4</sup>, \
Shichen Tang<sup>1</sup>, Lavdim Halilaj<sup>5</sup>, Juergen Luettin<sup>5</sup>, Hao Sun<sup>1</sup>

<sup>1</sup>Bosch Corporate Research RIX  
<sup>2</sup>Shanghai Jiaotong University 
<sup>3</sup>Shanghai University \
<sup>4</sup>AIR, Tsinghua University
<sup>5</sup>Robert Bosch GmbH

(\*) Equal contribution. (†) Corresponding author.  

<a href="https://www.arxiv.org/pdf/2508.01778"><img src='https://img.shields.io/badge/arXiv-DiffSemanticFusion-blue' alt='Paper PDF'></a>
<a href="https://www.arxiv.org/pdf/2508.01778"><img src='https://img.shields.io/badge/Datasets-MaplessQCNet-red' alt='Datasets'></a>
<a href="https://www.arxiv.org/pdf/2508.01778"><img src='https://img.shields.io/badge/Datasets-Sparse4D-green' alt='Datasets'></a>

</div>


## Overview

![github_diffsemanticfusion](https://github.com/user-attachments/assets/8bba0baa-252b-4be9-af3c-26f92c9f2f9b)

## Abstract         

Autonomous driving requires accurate scene understanding, including road geometry, traffic agents, and their semantic relationships. In online HD map generation scenarios, raster-based representations are well-suited to vision models but lack geometric precision, while graph-based representations retain structural detail but become unstable without precise maps. To harness the complementary strengths of both, we propose DiffSemanticFusion—a fusion framework for multimodal trajectory prediction and planning. Our approach reasons over a semantic raster–fused BEV space, enhanced by a map diffusion module that improves both the stability and expressiveness of online HD map representations. We validate our framework on two downstream tasks: trajectory prediction and planning-oriented end-to-end autonomous driving. Experiments on real-world autonomous driving benchmarks, nuScenes and NAVSIM, demonstrate improved performance over several state-of-the-art (SOTA) methods. For the prediction task on nuScenes, we integrate DiffSemanticFusion with the online HD map informed QCNet, achieving a 5.1\% performance improvement. For end-to-end autonomous driving in NAVSIM, DiffSemanticFusion achieves SOTA results, with a 15\% performance gain in NavHard scenarios. In addition, extensive ablation and sensitivity studies show that our map diffusion module can be seamlessly integrated into other vector-based approaches to enhance performance. 



## News
`[2025/08/06]` [ArXiv](https://www.arxiv.org/pdf/2508.01778) paper release. Code/Models are coming soon. Please stay tuned! ☕️\
`[2025/08/07]` Open source the mapless QCNet, as we train the model on GPU Cluster, also with the stdout and stderr
`[2025/08/19]` Provide quite detail training and eval process of Mapless QCNet, as shown in computing_jobs/qc_net_mapless_prediction_train.8564846.stdout




## Updates
We are going to release code step by step:

- [x] Mapless QCNet 
- [x] Mapless QCnet with Online HD Map Diffusion
- [ ] DiffSemanticFusion Base
- [ ] DiffSemanticFusion + Sparse4D Sparse
- [ ] DiffSemanticFusion + Sparse Graph
- [ ] DiffSemanticFusion

Note: Due to policy, SemanticFormer can't be open source, so we only open source homogeneous graph fusion with BEV

Note: Code needs to be cleaned and I will open source all the code within one month. As I promised.

## 📄 Citation

If you find DiffSemanticFusion is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@misc{sun2025diffsemanticfusionsemanticrasterbev,
      title={DiffSemanticFusion: Semantic Raster BEV Fusion for Autonomous Driving via Online HD Map Diffusion}, 
      author={Zhigang Sun and Yiru Wang and Anqing Jiang and Shuo Wang and Yu Gao and Yuwen Heng and Shouyi Zhang and An He and Hao Jiang and Jinhao Chai and Zichong Gu and Wang Jijun and Shichen Tang and Lavdim Halilaj and Juergen Luettin and Hao Sun},
      year={2025},
      eprint={2508.01778},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.01778}, 
}
```

```bibtex
@article{sun2024semanticformer,
  title={Semanticformer: Holistic and semantic traffic scene representation for trajectory prediction using knowledge graphs},
  author={Sun, Zhigang and Wang, Zixu and Halilaj, Lavdim and Luettin, Juergen},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

