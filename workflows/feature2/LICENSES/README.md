# Third-Party Model & Dataset Licenses

This directory contains the licenses for all third-party models, libraries, and datasets used in this project.

## License Summary

| Model/Library/Dataset | License | File |
|----------------------|---------|------|
| MiDaS (Depth Estimation) | MIT | [LICENSE-MiDaS.txt](LICENSE-MiDaS.txt) |
| VIDIT Dataset | CC BY-NC-SA 4.0 | [LICENSE-VIDIT.txt](LICENSE-VIDIT.txt) |

---

## Model Usage

### MiDaS (Intel ISL)

MiDaS is used for monocular depth estimation in this project. It provides:
- Depth maps from single RGB images
- Zero-shot cross-dataset generalization
- Fast inference suitable for real-time applications

The model is loaded automatically via PyTorch Hub:
```python
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
```

**Links:**
- GitHub: https://github.com/isl-org/MiDaS
- Paper: [Towards Robust Monocular Depth Estimation](https://arxiv.org/abs/1907.01341)

**Citation:**
```bibtex
@article{ranftl2020midas,
  title={Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author={Ranftl, Ren{\'e} and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```

---

## Dataset Usage

### VIDIT — Virtual Image Dataset for Illumination Transfer

VIDIT is used to train the TinyRelightNet model. It provides:
- Synthetic scenes with controllable lighting
- 8 light directions (N, S, E, W, NE, NW, SE, SW)
- Ground truth depth maps
- Multiple color temperatures

The dataset is loaded via Hugging Face:
```python
from datasets import load_dataset
hf_ds = load_dataset("Nahrawy/VIDIT-Depth-ControlNet")
```

**Links:**
- Hugging Face: https://huggingface.co/datasets/Nahrawy/VIDIT-Depth-ControlNet
- Original GitHub: https://github.com/majedelhelou/VIDIT
- Paper: [VIDIT: Virtual Image Dataset for Illumination Transfer](https://arxiv.org/abs/2005.05460)

**Citation:**
```bibtex
@inproceedings{helou2020vidit,
  title={VIDIT: Virtual Image Dataset for Illumination Transfer},
  author={El Helou, Majed and Zhou, Ruofan and Barthas, Johan and S{\"u}sstrunk, Sabine},
  booktitle={arXiv preprint arXiv:2005.05460},
  year={2020}
}
```

---

## Attribution

When using this project, please include attribution to the original model and dataset authors as specified in each license file.

## Contact

For questions about licensing, please refer to the original project repositories:
- MiDaS: https://github.com/isl-org/MiDaS
- VIDIT: https://github.com/majedelhelou/VIDIT
