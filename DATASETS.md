# Datasets

The datasets are **not included** in this repository. RAF-DB and RAF-AU are released by PRIS (Beijing University of Posts and Telecommunications) for **non-commercial research only**, and the licence does not allow copying, publishing or redistributing any part of them. Request access yourself through the official page: <http://www.whdeng.cn/RAF/model1.html> (an institutional email address is required).

## Expected layout

```text
TestDb/
└── DATASET/
    ├── train/            RAF-DB basic emotions, aligned 100×100 faces
    │   ├── 1/  Surprise  (1,290)
    │   ├── 2/  Fear      (281)
    │   ├── 3/  Disgust   (717)
    │   ├── 4/  Happiness (4,772)
    │   ├── 5/  Sadness   (1,982)
    │   ├── 6/  Anger     (705)
    │   └── 7/  Neutral   (2,524)
    └── test/             same folders: 329 / 74 / 160 / 1,185 / 478 / 162 / 680  (3,068 in total)

RAF-AU-<download>/RAF-AU/
├── aligned/              <id>_aligned.jpg   (4,601 faces)
└── RAFAU_label.txt       "<id>.jpg <AU>+<AU>+..."  or  "null"
```

Folder numbers follow the RAF-DB convention: **7 = Neutral** (there is no Contempt class in RAF-DB basic emotions).

The web app only needs `TestDb/DATASET/test` for its sample buttons and the batch report; prediction on uploaded images works without any dataset.

## Weights

- `models/resnet50_ft_weight.pkl` — VGGFace2 ResNet-50 (`resnet50_ft`), needed only for training the emotion engine. Download it from the "Pretrained models" table of <https://github.com/cydonia999/VGGFace2-pytorch>.
- The InceptionResnetV1 VGGFace2 weights used by `train_rafau.py` are downloaded automatically by `facenet-pytorch`.
