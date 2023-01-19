# Three Frameworks: MTL-VF, IMTL-AGF, PG-GS
Pytorch implementation for the paper "Towards Accurate and Interpretable Surgical Skill Assessment: A Video-Based Method Incorporating Recognized Surgical Gestures and Skill Levels" published at MICCAI 2020 and its expansion work published at IJCARS.

## Dataset
* The features can be downloaded from the Google Drive [link]() and put in local folder **```./data/features/```**.
* The annotations for JIGSAWS dataset can be accessed online and put in local folder **```./data/```**.
* Additional annotations as stated in this paper can be accessed via email to us.

## Usage
```python
python3 main.py
```

## Citation
```
@inproceedings{wang2020towards,
  title={Towards accurate and interpretable surgical skill assessment: A video-based method incorporating recognized surgical gestures and skill levels},
  author={Wang, Tianyu and Wang, Yijie and Li, Mian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={668--678},
  year={2020},
  organization={Springer}
}

@article{wang2021towards,
  title={Towards accurate and interpretable surgical skill assessment: a video-based method for skill score prediction and guiding feedback generation},
  author={Wang, Tianyu and Jin, Minhao and Li, Mian},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={16},
  number={9},
  pages={1595--1605},
  year={2021},
  publisher={Springer}
}
```
