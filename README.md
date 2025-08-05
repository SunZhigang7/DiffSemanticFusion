# DiffSemanticFusion
![github_diffsemanticfusion](https://github.com/user-attachments/assets/8bba0baa-252b-4be9-af3c-26f92c9f2f9b)

This is the official release of DiffSemanticFusion, Our approach reasons over a semantic raster–fused BEV space, enhanced by a map diffusion module that improves both the stability and expressiveness of online HD map representations.

We are going to release code step by step:

- [x] Mapless QCNet 
- [x] Mapless QCnet with Online HD Map Diffusion
- [ ] DiffSemanticFusion Base
- [ ] DiffSemanticFusion + Sparse4D Sparse
- [ ] DiffSemanticFusion + Sparse Graph
- [ ] DiffSemanticFusion

Note: Due to policy, SemanticFormer can't be open source, so we only open source homogeneous graph fusion with BEV

If our work is helpful, please consider cite:

## 📄 Citation

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

