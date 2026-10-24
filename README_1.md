# Raw data are not stored in this repo

The source dataset has no stated license and was compiled from The Numbers, IMDb, Rotten Tomatoes and Wikipedia, so the data files are not included.

**Source:** Kaggle dataset `monkeybusiness7/marvel-cinematic-universe-box-office` (https://www.kaggle.com/datasets/monkeybusiness7/marvel-cinematic-universe-box-office)

To reproduce:
1. Download the dataset from the Kaggle page, or in Python:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("monkeybusiness7/marvel-cinematic-universe-box-office")
   print(path)  # folder containing the downloaded file(s)
   ```
   (`pip install kagglehub` first.)
2. Copy the CSV from that folder into this `data/` folder and name it `marvel_box_office.csv`.
3. From the repo root, run `python code/analysis.py`.
