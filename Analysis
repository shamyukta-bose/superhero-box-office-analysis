"""
What drives superhero film box office? Cleaning + analysis.

Run from the repo root:  python code/analysis.py
Expects the Kaggle CSV at data/marvel_box_office.csv (see data/README.md).
Writes: data/marvel_box_office_cleaned.csv and results/analysis_results.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "marvel_box_office.csv"
CLEAN = ROOT / "data" / "marvel_box_office_cleaned.csv"
RESULTS = ROOT / "results" / "analysis_results.txt"

# ---------------------------------------------------------------- cleaning
df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
df["Movie"] = df["Movie"].str.strip().replace({
    "Guardians of The Galaxy": "Guardians of the Galaxy",
    "Avengers Infinity War": "Avengers: Infinity War",
    "Venom: Let There be Carnage": "Venom: Let There Be Carnage",
})
df["Release Date"] = pd.to_datetime(df["Release Date"], format="%m/%d/%Y")
df = df.drop_duplicates("Movie")

month = df["Release Date"].dt.month
df["release_month_num"] = month
df["release_season"] = np.select(
    [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
    ["Winter", "Spring", "Summer"], "Fall")
df["studio_group"] = df["Ownership"].where(
    df["Ownership"].isin(["Marvel Studios", "20th Century Fox", "Sony Pictures"]), "Other")
df["sony_flag"] = (df["Ownership"] == "Sony Pictures").astype(int)
df["legs_multiplier"] = df["Worldwide Box Office"] / df["Opening Weekend"]  # worldwide / opening
df["roi_multiple"] = df["Inflation Adjusted Worldwide"] / df["Inflation Adjusted Budget"]
df["mcu_flag"] = df["Phase"].notna().astype(int)

# sequel_flag: 0 = first film of its franchise/series, 1 = any later entry
# (sequel, prequel, reboot, spin-off). Judgment calls are listed in the README.
FIRST_IN_FRANCHISE = ["Iron Man", "Spider-Man", "X-Men", "Fantastic Four (2005)", "Daredevil",
                      "Blade", "Ghost Rider", "Hulk", "The Punisher", "Venom"]
df["sequel_flag"] = (~df["Movie"].isin(FIRST_IN_FRANCHISE)).astype(int)

clean = df.rename(columns={
    "Movie": "title", "Release Date": "release_date", "Release Year": "release_year",
    "Ownership": "studio", "Domestic Box Office": "domestic_gross",
    "International Box Office": "international_gross", "Worldwide Box Office": "worldwide_gross",
    "Opening Weekend": "opening_weekend", "Budget": "budget",
    "Inflation Adjusted Worldwide": "worldwide_gross_adj",
    "Inflation Adjusted Opening Weekend": "opening_weekend_adj",
    "Inflation Adjusted Budget": "budget_adj", "IMDb Score": "imdb_score",
    "Meta Score": "meta_score", "Tomatometer": "tomatometer",
    "Rotten Tomato Audience Score": "audience_score", "Run Time In Minutes": "runtime_min",
    "Phase": "mcu_phase", "Director": "director"})
keep = ["title", "release_date", "release_year", "release_month_num", "release_season", "studio",
        "studio_group", "sony_flag", "mcu_flag", "mcu_phase", "sequel_flag", "opening_weekend",
        "opening_weekend_adj", "domestic_gross", "international_gross", "worldwide_gross",
        "worldwide_gross_adj", "budget", "budget_adj", "legs_multiplier", "roi_multiple",
        "tomatometer", "audience_score", "meta_score", "imdb_score", "runtime_min", "director"]
clean[keep].to_csv(CLEAN, index=False)

out = []
P = out.append
P(f"Films: {len(clean)} | years {clean.release_year.min()}-{clean.release_year.max()}")
P(f"Missing values per column (non-zero only):\n{clean[keep].isna().sum()[lambda s: s > 0].to_string()}")

# ------------------------------------------------ Q1: timing (legs by season)
P("\nQ1: LEGS MULTIPLIER (worldwide / opening weekend) BY SEASON")
P(clean.groupby("release_season")["legs_multiplier"].agg(["count", "median", "mean"]).round(2).to_string())
groups = [x["legs_multiplier"].values for _, x in clean.groupby("release_season")]
P(f"Kruskal-Wallis p = {stats.kruskal(*groups).pvalue:.3f}")
may = clean[clean.release_month_num == 5]["legs_multiplier"]
oth = clean[clean.release_month_num != 5]["legs_multiplier"]
P(f"May median {may.median():.2f} (n={len(may)}) vs other months {oth.median():.2f} (n={len(oth)}), "
  f"Mann-Whitney p = {stats.mannwhitneyu(may, oth).pvalue:.3f}")
P("\nMedian worldwide gross (inflation-adjusted, $M) by season:")
P((clean.groupby("release_season")["worldwide_gross_adj"].median() / 1e6).round(0).to_string())

# ------------------------------------------- descriptive checks for the charts
P("\nMedian worldwide gross (inflation-adjusted, $M) by release month (n in brackets):")
bym = clean.groupby("release_month_num")["worldwide_gross_adj"].agg(["count", "median"])
bym["median"] = (bym["median"] / 1e6).round(0)
P(bym.to_string())
P("\nSequel flag (0 = first in franchise, 1 = later entry): median worldwide gross, inflation-adjusted, $M")
bys = clean.groupby("sequel_flag")["worldwide_gross_adj"].agg(["count", "median"])
bys["median"] = (bys["median"] / 1e6).round(0)
P(bys.to_string())

# --------------------------------------------------------- Q2: regression
d = clean.copy()
d["ln_ww"] = np.log(d["worldwide_gross_adj"])
d["ln_bud"] = np.log(d["budget_adj"])
d["tomato"] = d["tomatometer"]
d["runtime"] = d["runtime_min"]
d["summer_may"] = d["release_month_num"].isin([5, 6, 7]).astype(int)
SONY = 'C(studio_group, Treatment("Marvel Studios"))[T.Sony Pictures]'
f = "ln_ww ~ ln_bud + tomato + runtime + summer_may + C(studio_group, Treatment(\"Marvel Studios\"))"
m = smf.ols(f, data=d).fit()
P("\nQ2: OLS ln(worldwide gross, inflation-adj) ~ ln(budget adj) + Tomatometer + runtime "
  "+ May-July release + studio group (ref: Marvel Studios)")
P(f"n={int(m.nobs)} R2={m.rsquared:.3f} adj R2={m.rsquared_adj:.3f} F p={m.f_pvalue:.4f}")
tab = pd.DataFrame({"coef": m.params, "p": m.pvalues,
                    "ci_low": m.conf_int()[0], "ci_high": m.conf_int()[1]}).round(3)
P(tab.to_string())
P(f"\nPlain-language effects: +10 Tomatometer points = {(np.exp(10 * m.params['tomato']) - 1) * 100:.1f}% "
  f"higher gross; +10% budget = {(1.1 ** m.params['ln_bud'] - 1) * 100:.1f}% higher gross; "
  f"Sony vs Marvel Studios = {(np.exp(m.params[SONY]) - 1) * 100:.0f}% "
  f"(CI {(np.exp(m.conf_int().loc[SONY, 0]) - 1) * 100:.0f}% to {(np.exp(m.conf_int().loc[SONY, 1]) - 1) * 100:.0f}%)")
X = m.model.exog
vif = pd.Series([variance_inflation_factor(X, i) for i in range(1, X.shape[1])],
                index=m.model.exog_names[1:]).round(1)
P("\nVIF:")
P(vif.to_string())

# robustness: robust standard errors and influential films
r = smf.ols(f, data=d).fit(cov_type="HC3")
P("\nRobust (HC3) p-values:")
P(r.pvalues.round(3).to_string())
inf = m.get_influence()
d["cooks"] = inf.cooks_distance[0]
d["stud_resid"] = inf.resid_studentized_external
cut = 4 / len(d)
P(f"\nMost influential films (Cook's distance; rule-of-thumb cutoff 4/n = {cut:.3f}):")
P(d.nlargest(5, "cooks")[["title", "studio_group", "cooks", "stud_resid"]].round(3).to_string(index=False))
d2 = d[d.cooks < cut]
m2 = smf.ols(f, data=d2).fit()
P(f"\nSensitivity: dropping the {len(d) - len(d2)} most influential films -> "
  f"Sony coef {m2.params[SONY]:.3f}, p = {m2.pvalues[SONY]:.3f}; R2 {m2.rsquared:.3f}")

# ------------------------------------------------------- Q3: studio / Sony
P("\nQ3: STUDIO COMPARISON (inflation-adjusted)")
s = clean.groupby("studio_group").agg(
    films=("title", "count"),
    median_ww_M=("worldwide_gross_adj", lambda x: round(x.median() / 1e6)),
    median_roi=("roi_multiple", "median"),
    median_tomato=("tomatometer", "median")).round(2)
P(s.to_string())
P("\nSony Pictures titles:")
sony = clean[clean.sony_flag == 1]
P(sony[["title", "release_year", "worldwide_gross_adj", "roi_multiple", "tomatometer"]]
  .assign(worldwide_gross_adj=lambda x: (x.worldwide_gross_adj / 1e6).round(0))
  .round(2).sort_values("roi_multiple", ascending=False).to_string(index=False))
rho = stats.spearmanr(sony["tomatometer"], sony["roi_multiple"])
P(f"\nSony films: Spearman corr (Tomatometer vs ROI) = {rho.correlation:.2f}, "
  f"p = {rho.pvalue:.3f} (n={len(sony)})")

RESULTS.write_text("\n".join(out))
print("\n".join(out))
