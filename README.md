# What drives superhero film box office?

A portfolio analysis of 68 live-action Marvel-character films (1998–2024), built with a focus on how studio, reviews, budget and release timing relate to box office, with a spotlight on Sony Pictures.

**Dashboard:** [Tableau Public](https://public.tableau.com/app/profile/shamyukta.bose/viz/AnalysisofboxofficeofSony/FinalAnalysis)

![Dashboard](tableau/dashboard.png)

## Key findings
- **Budget and critic reviews go with higher gross.** Holding other factors constant, each +10 Tomatometer points is associated with roughly 18% higher worldwide gross, and a 10% larger budget with roughly 8% higher gross.
- **Release timing shows no clear effect.** Legs (worldwide gross / opening weekend) does not differ significantly by season (Kruskal-Wallis p = 0.30), and a May–July release window was not significant in the regression.
- **Sony's returns vary widely**, from Venom (7.4x budget) to Madame Web (1.2x). Better-reviewed Sony films tend to return more (rank correlation 0.65, n = 11).
- **Sony films earned about 48% more than comparable Marvel Studios films** (p = 0.046), but the range is wide (about +1% to +118%) and the result is sensitive to a few films (it is not significant once the six most influential films are removed).

## Method
- Cleaned and standardized the Kaggle data, then added season, legs multiplier, return multiple and studio-group fields.
- OLS regression in Python (statsmodels): log of inflation-adjusted worldwide gross on log budget, Tomatometer, runtime, a May–July release flag and studio group (reference: Marvel Studios). R² = 0.76, adjusted R² = 0.73.
- Checks: variance inflation factors, robust (HC3) standard errors, and Cook's distance for influential films.
- Return = worldwide gross / budget (not profit). Legs = worldwide gross / opening weekend.

## Limitations
- **Association, not causation.** Marketing spend, competition and franchise strength are not in the data.
- **Small sample:** 68 films, 11 of them Sony-owned. The three Sony-distributed Spider-Man MCU films are grouped under Marvel Studios in the source data.
- **Scope:** superhero films only, so results may not apply to other genres.
- Films within a franchise are not independent, which the model does not account for.

## Reproduce
The data are not included: the Kaggle dataset has no stated license and was compiled from The Numbers, IMDb, Rotten Tomatoes and Wikipedia.
1. Download the Kaggle dataset `monkeybusiness7/marvel-cinematic-universe-box-office` (https://www.kaggle.com/datasets/monkeybusiness7/marvel-cinematic-universe-box-office) and save it as `data/marvel_box_office.csv`.
2. `pip install -r requirements.txt`
3. `python code/analysis.py` writes the cleaned file to `data/` and the results to `results/analysis_results.txt`.

## Repo structure
```
code/analysis.py      cleaning + analysis
results/              script output
tableau/              dashboard screenshot
data/README.md        how to get the data
```

**Data credit:** Kaggle dataset `marvel-cinematic-universe-box-office` by monkeybusiness7, compiled from The Numbers (box office), IMDb and Rotten Tomatoes (ratings) and Wikipedia. Inflation-adjusted figures are the dataset's own.
