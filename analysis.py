#%% SETUP
# Applied Data Science project - wildlife-trafficking marketplace listings (VIDA-NYU/LTS)
# Run:  DATA_DIR=path/to/data_use_cases python analysis.py
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import TTestIndPower
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (f1_score, average_precision_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_curve)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
SEED = 42; rng = np.random.default_rng(SEED); np.random.seed(SEED)
DATA = os.environ.get("DATA_DIR", "data_use_cases"); FIG = "figures"; os.makedirs(FIG, exist_ok=True)
PAL = dict(plum="#5D4560", teal="#1A87A3", pink="#DC6578", brown="#8B3A1E", orange="#D96030",
           olive="#6F7B22", lime="#A8B820", gold="#D4A62A", cream="#FFF4EE")
plt.rcParams.update({"figure.facecolor": PAL["cream"], "axes.facecolor": PAL["cream"], "savefig.facecolor": PAL["cream"],
                     "axes.edgecolor": PAL["plum"], "axes.labelcolor": PAL["plum"], "text.color": PAL["plum"],
                     "xtick.color": PAL["plum"], "ytick.color": PAL["plum"], "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
def save(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=110); plt.close()
R = {}  # all results are collected here and written to results.json

sh = pd.read_csv(f"{DATA}/shark_trophy.csv")
lea = pd.read_csv(f"{DATA}/data_leather.csv", low_memory=False)
lval = pd.read_csv(f"{DATA}/leather_validation.csv")
sval = pd.read_csv(f"{DATA}/validation_sharks.csv")

#%% M1 Data science process, preparation, model building, validation
sh["title_clean"] = sh.title.str.replace(r"\s*\|\s*eBay$", "", regex=True).str.strip()
sh["t"] = pd.to_datetime(sh.retrieved, utc=True)
sh["country"] = sh.location.fillna("Unknown").str.split(",").str[-1].str.strip()
sh["is_us"] = (sh.country == "United States").astype(int)
sh["title_len"] = sh.title_clean.str.len(); sh["n_words"] = sh.title_clean.str.split().str.len()
sh["upper_ratio"] = sh.title_clean.apply(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)))
sh["has_digit"] = sh.title_clean.str.contains(r"\d").astype(int)
sh["desc_len"] = sh.description.fillna("").str.len(); sh["log_desc"] = np.log1p(sh.desc_len)
sh["log_price"] = np.log(sh.price)
low = sh.title_clean.str.lower()
sh["kw_jaw"] = low.str.contains("jaw").astype(int); sh["kw_tooth"] = low.str.contains("tooth|teeth").astype(int)
sh["kw_fossil"] = low.str.contains("fossil|megalodon").astype(int)
dup_title = int(sh.title_clean.duplicated().sum())
conflict = int((sh.groupby("title_clean").label.nunique() > 1).sum())
R["m1"] = dict(rows=len(sh), cols=sh.shape[1] - 0, leather_rows=len(lea), leather_cols=lea.shape[1],
               shark_val_rows=len(sval), leather_val_rows=len(lval),
               dtypes_numeric=int(sh.select_dtypes("number").shape[1]), missing=sh[["image", "location"]].isna().sum().to_dict(),
               dup_urls=int(sh.url.duplicated().sum()), dup_titles=dup_title, conflicting_dup_titles=conflict,
               pos=int(sh.label.sum()), pos_rate=float(sh.label.mean()),
               t_min=str(sh.t.min()), t_max=str(sh.t.max()), sellers=int(sh.seller.nunique()),
               memory_mb=float(sh.memory_usage(deep=True).sum() / 1e6), leather_memory_mb=float(lea.memory_usage(deep=True).sum() / 1e6))
# modelling copy: de-duplicate titles so identical listings cannot leak between folds
ml = sh.drop_duplicates("title_clean").reset_index(drop=True)
R["m1"]["ml_rows"] = len(ml); R["m1"]["ml_pos"] = int(ml.label.sum())
def text_lr(cw=None): return Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=20000)),
                                       ("lr", LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED))])
X, y = ml.title_clean, ml.label.values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv = cross_validate(text_lr(), X, y, cv=skf, scoring=["f1", "average_precision", "precision", "recall"])
R["m1"]["kfold"] = {k: [float(v) for v in cv[f"test_{k}"]] for k in ["f1", "average_precision", "precision", "recall"]}
# Bootstrap: refit on resampled training rows, score on out-of-bag rows (B=100)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xtr = Xtr.reset_index(drop=True); boot = []
for b in range(100):
    idx = rng.integers(0, len(Xtr), len(Xtr)); oob = np.setdiff1d(np.arange(len(Xtr)), idx)
    m = text_lr().fit(Xtr.iloc[idx], ytr[idx]); boot.append(f1_score(ytr[oob], m.predict(Xtr.iloc[oob])))
R["m1"]["boot_f1"] = dict(mean=float(np.mean(boot)), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), B=100)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(range(1, 6), R["m1"]["kfold"]["f1"], color=PAL["teal"]); ax[0].axhline(np.mean(R["m1"]["kfold"]["f1"]), color=PAL["orange"], ls="--")
ax[0].set(title="5-fold stratified CV: F1 per fold", xlabel="fold", ylabel="F1", ylim=(0, 1))
ax[1].hist(boot, bins=18, color=PAL["pink"]); ax[1].set(title="Bootstrap (B=100) out-of-bag F1", xlabel="F1")
save("m1_validation")

#%% M2 Descriptive statistics and visual analytics
def describe(s):
    s = pd.Series(s).dropna(); q1, q2, q3 = s.quantile([.25, .5, .75])
    sd = s.std()
    return dict(n=int(len(s)), mean=s.mean(), median=q2, mode=float(s.round(2).mode().iloc[0]), std=sd, var=s.var(), min=s.min(), max=s.max(),
                range=s.max() - s.min(), q1=q1, q3=q3, iqr=q3 - q1, cv=sd / s.mean(),
                pearson_skew=3 * (s.mean() - q2) / sd,                     # Karl Pearson's 2nd coefficient
                bowley=(q3 + q1 - 2 * q2) / (q3 - q1),                      # Bowley's quartile coefficient
                moment_skew=stats.skew(s), excess_kurtosis=stats.kurtosis(s))
R["m2"] = {c: {k: float(v) for k, v in describe(sh[c]).items()} for c in ["price", "log_price", "title_len", "n_words", "desc_len"]}
num = ["log_price", "title_len", "n_words", "upper_ratio", "log_desc", "is_us", "label"]
corr = sh[num].corr(); R["m2"]["corr"] = corr.round(3).to_dict()
r, p = stats.pearsonr(sh.title_len, sh.log_price); rs, ps = stats.spearmanr(sh.title_len, sh.log_price)
R["m2"]["title_len_vs_logprice"] = dict(pearson=r, p=p, spearman=rs, p_s=ps)
R["m2"]["by_label"] = sh.groupby("label").agg(n=("price", "size"), median_price=("price", "median"), mean_price=("price", "mean"),
                                              median_len=("title_len", "median")).round(2).reset_index().to_dict("records")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].hist(sh.price.clip(upper=sh.price.quantile(.99)), bins=50, color=PAL["orange"]); ax[0].set(title="Price (clipped at 99th pct)", xlabel="USD")
sns.histplot(sh.log_price, bins=50, kde=True, color=PAL["teal"], ax=ax[1]); ax[1].set(title="log(price)", xlabel="ln USD")
sns.histplot(sh.title_len, bins=40, color=PAL["pink"], ax=ax[2]); ax[2].set(title="Title length", xlabel="characters")
save("m2_hist")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
sns.violinplot(data=sh, x="label", y="log_price", palette=[PAL["teal"], PAL["orange"]], ax=ax[0]); ax[0].set_title("log(price) by label (violin)")
sns.boxplot(data=sh, x="label", y="title_len", palette=[PAL["teal"], PAL["orange"]], ax=ax[1]); ax[1].set_title("Title length by label (box)")
sns.boxenplot(data=sh, x="label", y="log_desc", palette=[PAL["teal"], PAL["orange"]], ax=ax[2]); ax[2].set_title("log(1+description length) (boxen)")
save("m2_violin_box")
s = sh.sample(4000, random_state=SEED)
plt.figure(figsize=(6.5, 4.2)); sns.scatterplot(data=s, x="title_len", y="log_price", hue="label", palette=[PAL["teal"], PAL["orange"]], alpha=.55, s=14)
plt.title("Title length vs log(price)"); save("m2_scatter")
g = sns.pairplot(sh.sample(1500, random_state=SEED)[["log_price", "title_len", "n_words", "log_desc", "label"]], hue="label",
                 palette=[PAL["teal"], PAL["orange"]], plot_kws=dict(alpha=.5, s=10), height=1.7)
g.figure.savefig(f"{FIG}/m2_pairplot.png", dpi=100); plt.close("all")
plt.figure(figsize=(6, 4.6)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuOr_r", center=0, square=True, cbar=False); plt.title("Pearson correlation heatmap")
save("m2_heatmap")
top = sh.country.value_counts().head(8).index
b = sh[sh.country.isin(top)].groupby("country").agg(n=("price", "size"), med=("price", "median"), pos=("label", "sum")).reset_index()
b["pos_rate"] = b.pos / b.n; R["m2"]["countries"] = b.round(3).to_dict("records")
plt.figure(figsize=(7.5, 4.4)); plt.scatter(b.n, b.med, s=b.pos_rate * 9000 + 40, c=PAL["pink"], alpha=.6, edgecolor=PAL["plum"])
for _, rw in b.iterrows(): plt.annotate(rw.country, (rw.n, rw.med), ha="center", fontsize=8)
plt.xscale("log"); plt.xlabel("listings (log scale)"); plt.ylabel("median price (USD)"); plt.title("Bubble chart: size = share of positive listings"); save("m2_bubble")

#%% M3 Inferential statistics
lp = sh.log_price
samp = lp.sample(5000, random_state=SEED)
R["m3"] = dict(shapiro_W=float(stats.shapiro(samp)[0]), shapiro_p=float(stats.shapiro(samp)[1]),
               dagostino_p=float(stats.normaltest(lp)[1]), mu=float(lp.mean()), sigma=float(lp.std()))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8)); stats.probplot(lp, plot=ax[0]); ax[0].set_title("Q-Q plot: log(price)")
xs = np.linspace(lp.min(), lp.max(), 200); ax[1].hist(lp, bins=50, density=True, color=PAL["teal"], alpha=.7)
ax[1].plot(xs, stats.norm.pdf(xs, lp.mean(), lp.std()), color=PAL["orange"], lw=2); ax[1].set_title("Normal fit to log(price)"); save("m3_normal")
# Central Limit Theorem on the (very right-skewed) raw price
clt = {}; fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
ax[0].hist(sh.price.clip(upper=500), bins=50, color=PAL["orange"]); ax[0].set_title("Population (clip 500)")
for a, n in zip(ax[1:], [5, 30, 300]):
    means = np.array([rng.choice(sh.price.values, n).mean() for _ in range(3000)]); clt[n] = dict(skew=float(stats.skew(means)), sd=float(means.std()))
    a.hist(means.clip(max=np.percentile(means, 99.5)), bins=40, color=PAL["teal"]); a.set_title(f"Sample means, n={n}")
R["m3"]["clt"] = clt; R["m3"]["pop_skew"] = float(stats.skew(sh.price)); save("m3_clt")
# Confidence intervals
def tci(x): m, se = x.mean(), stats.sem(x); lo, hi = stats.t.interval(.95, len(x) - 1, m, se); return dict(mean=float(m), lo=float(lo), hi=float(hi))
R["m3"]["ci_price"] = tci(sh.price); R["m3"]["ci_logprice"] = tci(lp)
R["m3"]["ci_geo_price"] = {k: float(np.exp(v)) for k, v in R["m3"]["ci_logprice"].items()}
lo, hi = proportion_confint(sh.label.sum(), len(sh), method="wilson"); R["m3"]["ci_pos_rate"] = dict(p=float(sh.label.mean()), lo=float(lo), hi=float(hi))
# Z-test (two proportions): share of positives, US vs non-US
cnt = sh.groupby("is_us").label.agg(["sum", "count"]); z, pz = proportions_ztest(cnt["sum"].values, cnt["count"].values)
R["m3"]["ztest"] = dict(z=float(z), p=float(pz), p_nonus=float(cnt.loc[0, "sum"] / cnt.loc[0, "count"]), p_us=float(cnt.loc[1, "sum"] / cnt.loc[1, "count"]),
                        n_nonus=int(cnt.loc[0, "count"]), n_us=int(cnt.loc[1, "count"]))
# Welch t-test: log price, positive vs negative listings
a1, a0 = lp[sh.label == 1], lp[sh.label == 0]; tt = stats.ttest_ind(a1, a0, equal_var=False)
pooled = np.sqrt(((len(a1) - 1) * a1.var() + (len(a0) - 1) * a0.var()) / (len(a1) + len(a0) - 2)); d = (a1.mean() - a0.mean()) / pooled
R["m3"]["ttest"] = dict(t=float(tt.statistic), p=float(tt.pvalue), df=float(tt.df), mean1=float(a1.mean()), mean0=float(a0.mean()), cohen_d=float(d))
# One-way ANOVA: log price across top-5 countries
t5 = sh.country.value_counts().head(5).index; groups = [lp[sh.country == c] for c in t5]
F, pF = stats.f_oneway(*groups); H, pH = stats.kruskal(*groups); lev = stats.levene(*groups)
ssb = sum(len(g) * (g.mean() - lp[sh.country.isin(t5)].mean()) ** 2 for g in groups); sst = ((lp[sh.country.isin(t5)] - lp[sh.country.isin(t5)].mean()) ** 2).sum()
R["m3"]["anova"] = dict(countries=list(t5), F=float(F), p=float(pF), eta2=float(ssb / sst), kruskal_H=float(H), kruskal_p=float(pH), levene_p=float(lev.pvalue),
                        means={c: float(g.mean()) for c, g in zip(t5, groups)})
plt.figure(figsize=(7, 3.8)); sns.boxplot(data=sh[sh.country.isin(t5)], x="country", y="log_price", order=list(t5), palette="Set2"); plt.xticks(rotation=15); plt.title("ANOVA groups: log(price) by country"); save("m3_anova")
# Type-I error (permutation of labels) and Type-II error / power
perm_p = []
lab = sh.label.values; lv = lp.values
for _ in range(1000):
    pl = rng.permutation(lab); perm_p.append(stats.ttest_ind(lv[pl == 1], lv[pl == 0], equal_var=False).pvalue)
R["m3"]["type1_rate"] = float(np.mean(np.array(perm_p) < .05))
pw = TTestIndPower(); R["m3"]["power_observed"] = float(pw.power(abs(d), len(a1), .05, ratio=len(a0) / len(a1)))
R["m3"]["mde_d_80"] = float(pw.solve_power(None, len(a1), .05, .8, ratio=len(a0) / len(a1)))
# Poisson: hourly retrieval counts (see M6 for the series)
hc = sh.t.dt.floor("h").value_counts().sort_index(); full = hc.reindex(pd.date_range(hc.index.min(), hc.index.max(), freq="h"))
runs = (full.isna() != full.isna().shift()).cumsum(); best = max((g for _, g in full.groupby(runs) if not g.isna().iloc[0]), key=len)
ser = best.astype(float); lam = ser.mean()
R["m3"]["poisson"] = dict(n_hours=int(len(ser)), lam=float(lam), var=float(ser.var()), dispersion=float(ser.var() / lam))
R["m3"]["poisson"]["chi2_p"] = float(stats.chi2.sf((len(ser) - 1) * ser.var() / lam, len(ser) - 1))  # dispersion test

#%% M4 Regression analysis
tr, te = train_test_split(sh, test_size=.2, random_state=SEED)
forms = {"A: title_len": "log_price ~ title_len",
         "B: + text shape": "log_price ~ title_len + n_words + upper_ratio + has_digit",
         "C: + label, US, desc": "log_price ~ title_len + n_words + upper_ratio + has_digit + label + is_us + log_desc",
         "D: + keywords": "log_price ~ title_len + n_words + upper_ratio + has_digit + label + is_us + log_desc + kw_jaw + kw_tooth + kw_fossil"}
fits, rows = {}, []
for k, f in forms.items():
    m = smf.ols(f, tr).fit(); fits[k] = m; smear = np.mean(np.exp(m.resid)); pr = np.exp(m.predict(te)) * smear
    rows.append(dict(model=k, k=int(m.df_model), r2=m.rsquared, adj_r2=m.rsquared_adj, aic=m.aic, bic=m.bic, F=m.fvalue, F_p=m.f_pvalue,
                     test_r2_log=1 - ((te.log_price - m.predict(te)) ** 2).sum() / ((te.log_price - te.log_price.mean()) ** 2).sum(),
                     test_mae_usd=float(np.mean(np.abs(te.price - pr))), test_mape_pct=float(np.mean(np.abs((te.price - pr) / te.price)) * 100),
                     test_mae_log=float(np.mean(np.abs(te.log_price - m.predict(te))))))
R["m4"] = dict(table=rows, n_train=len(tr), n_test=len(te), baseline_mae_usd=float(np.mean(np.abs(te.price - tr.price.median()))),
               baseline_mape_pct=float(np.mean(np.abs((te.price - tr.price.median()) / te.price)) * 100))
best_k = min(rows, key=lambda r: r["aic"])["model"]; bm = fits[best_k]; R["m4"]["best"] = best_k
ci = bm.conf_int(); R["m4"]["coef"] = [dict(term=i, coef=float(bm.params[i]), se=float(bm.bse[i]), t=float(bm.tvalues[i]), p=float(bm.pvalues[i]),
                                          lo=float(ci.loc[i, 0]), hi=float(ci.loc[i, 1])) for i in bm.params.index]
R["m4"]["nested_F"] = {f"{a} -> {b_}": dict(F=float(anova_lm(fits[a], fits[b_]).iloc[1]["F"]), p=float(anova_lm(fits[a], fits[b_]).iloc[1]["Pr(>F)"]))
                       for a, b_ in zip(list(forms)[:-1], list(forms)[1:])}
R["m4"]["resid"] = dict(skew=float(stats.skew(bm.resid)), kurt=float(stats.kurtosis(bm.resid)), bp_p=float(sm.stats.diagnostic.het_breuschpagan(bm.resid, bm.model.exog)[1]),
                        dw=float(sm.stats.durbin_watson(bm.resid)))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].scatter(bm.fittedvalues, bm.resid, s=5, alpha=.3, color=PAL["teal"]); ax[0].axhline(0, color=PAL["orange"]); ax[0].set(title="Residuals vs fitted", xlabel="fitted", ylabel="residual")
stats.probplot(bm.resid, plot=ax[1]); ax[1].set_title("Residual Q-Q")
ax[2].scatter(te.log_price, bm.predict(te), s=5, alpha=.3, color=PAL["pink"]); ax[2].plot([0, 10], [0, 10], color=PAL["plum"]); ax[2].set(title="Test: actual vs predicted", xlabel="actual ln price", ylabel="predicted"); save("m4_residuals")
plt.figure(figsize=(7, 3.6)); mm = pd.DataFrame(rows); x = np.arange(len(mm)); plt.bar(x - .2, mm.adj_r2, .4, color=PAL["teal"], label="adj R²"); plt.bar(x + .2, mm.test_r2_log, .4, color=PAL["orange"], label="test R²")
plt.xticks(x, [r.split(":")[0] for r in mm.model]); plt.legend(); plt.title("Model comparison"); save("m4_compare")

#%% M5 Anomaly detection, outlier causes, SMOTE
feat = StandardScaler().fit_transform(sh[["log_price", "title_len", "n_words"]]); flags = {}
zs = (lp - lp.mean()) / lp.std(); flags["z-score |z|>3"] = (zs.abs() > 3).values
q1, q3 = lp.quantile([.25, .75]); flags["IQR fence (log price)"] = ((lp < q1 - 1.5 * (q3 - q1)) | (lp > q3 + 1.5 * (q3 - q1))).values
nn = NearestNeighbors(n_neighbors=11).fit(feat); dist = nn.kneighbors(feat)[0][:, 1:].mean(1); flags["kNN distance (top 1%)"] = dist > np.quantile(dist, .99)
flags["LOF (density, 1%)"] = LocalOutlierFactor(n_neighbors=20, contamination=.01).fit_predict(feat) == -1
flags["DBSCAN noise"] = DBSCAN(eps=.35, min_samples=10).fit_predict(feat) == -1
base = sh.label.mean(); R["m5"] = dict(methods={}, base_pos_rate=float(base))
for k, v in flags.items(): R["m5"]["methods"][k] = dict(n=int(v.sum()), pct=float(v.mean() * 100), pos_rate=float(sh.label[v].mean()) if v.sum() else None, lift=float(sh.label[v].mean() / base) if v.sum() else None)
names = list(flags); J = pd.DataFrame([[(flags[a] & flags[b]).sum() / max(1, (flags[a] | flags[b]).sum()) for b in names] for a in names], index=names, columns=names)
R["m5"]["jaccard"] = J.round(3).to_dict()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for a, k in zip(ax, ["z-score |z|>3", "kNN distance (top 1%)", "LOF (density, 1%)", "DBSCAN noise"]):
    a.scatter(sh.title_len, sh.log_price, s=3, color=PAL["teal"], alpha=.25); a.scatter(sh.title_len[flags[k]], sh.log_price[flags[k]], s=12, color=PAL["orange"]); a.set_title(f"{k}\n({flags[k].sum()} flagged)", fontsize=9); a.set_xlabel("title length")
ax[0].set_ylabel("ln price"); save("m5_methods")
R["m5"]["top_prices"] = sh.nlargest(6, "price")[["price", "title_clean"]].assign(title_clean=lambda d_: d_.title_clean.str[:70]).to_dict("records")
# Outlier causes in the (much larger) leather dataset
lp_ = lea.price.dropna(); lq1, lq3 = lp_.quantile([.25, .75])
R["m5"]["leather"] = dict(n=len(lea), zero=int((lea.price == 0).sum()), gt_1e5=int((lea.price > 1e5).sum()), gt_1e6=int((lea.price > 1e6).sum()), median=float(lp_.median()),
                          iqr_upper=float(lq3 + 1.5 * (lq3 - lq1)), iqr_out_pct=float((lp_ > lq3 + 1.5 * (lq3 - lq1)).mean() * 100), max=float(lp_.max()), currencies=lea.currency.value_counts().head(5).to_dict(),
                          top=lea.nlargest(5, "price")[["price", "currency", "title"]].assign(title=lambda d_: d_.title.str[:70]).to_dict("records"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); ax[0].hist(np.log10(lp_[lp_ > 0]), bins=60, color=PAL["brown"]); ax[0].set(title="Leather: log10(price) (n=%d)" % len(lp_), xlabel="log10 price")
sns.boxplot(x=np.log10(lp_[lp_ > 0]), color=PAL["gold"], ax=ax[1]); ax[1].set_title("Leather: box plot of log10(price)"); save("m5_leather")
# SMOTE vs class weights vs plain, 5-fold CV (SMOTE applied inside each training fold only)
tf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=20000)
cands = {"Plain LR": Pipeline([("tf", tf()), ("lr", LogisticRegression(max_iter=1000))]),
         "Class-weighted LR": Pipeline([("tf", tf()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
         "SMOTE + LR": ImbPipeline([("tf", tf()), ("sm", SMOTE(random_state=SEED)), ("lr", LogisticRegression(max_iter=1000))])}
R["m5"]["smote"] = {}; fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, pipe in cands.items():
    c = cross_validate(pipe, X, y, cv=skf, scoring=["f1", "average_precision", "precision", "recall"])
    pipe.fit(Xtr, ytr); pr = pipe.predict(Xte); sc = pipe.predict_proba(Xte)[:, 1]; tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    R["m5"]["smote"][k] = dict(cv_f1=float(c["test_f1"].mean()), cv_f1_sd=float(c["test_f1"].std()), cv_ap=float(c["test_average_precision"].mean()), cv_prec=float(c["test_precision"].mean()), cv_rec=float(c["test_recall"].mean()),
                               ho_f1=float(f1_score(yte, pr)), ho_prec=float(precision_score(yte, pr)), ho_rec=float(recall_score(yte, pr)), ho_ap=float(average_precision_score(yte, sc)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    p_, r_, _ = precision_recall_curve(yte, sc); ax[0].plot(r_, p_, label=k)
ax[0].set(title="Hold-out precision-recall curves", xlabel="recall", ylabel="precision"); ax[0].legend(fontsize=8)
kk = list(cands); ax[1].bar(kk, [R["m5"]["smote"][k]["cv_f1"] for k in kk], color=[PAL["teal"], PAL["olive"], PAL["pink"]]); ax[1].set(title="5-fold CV F1", ylim=(0, 1)); plt.setp(ax[1].get_xticklabels(), rotation=12, fontsize=8); save("m5_smote")
R["m5"]["pos_in_train"] = int(ytr.sum()); R["m5"]["n_train"] = len(ytr)
# GPT-labelled subset of the leather validation file as an evaluation-metrics example
g = lval.dropna(subset=["GPT_label"]); tn, fp, fn, tp = confusion_matrix(g.label, g.GPT_label.astype(int)).ravel()
R["m5"]["gpt_check"] = dict(n=len(g), acc=float((tn + tp) / len(g)), prec=float(tp / (tp + fp)), rec=float(tp / (tp + fn)), f1=float(2 * tp / (2 * tp + fp + fn)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                            val_pos=int(lval.label.sum()), val_n=len(lval))

#%% M6 Time series forecasting (hourly count of retrieved listings, longest gap-free crawl run)
ts = ser.copy(); ts.index = ts.index.tz_localize(None) if ts.index.tz is not None else ts.index
nte = 24; trn, tst = ts.iloc[:-nte], ts.iloc[-nte:]
adf = adfuller(trn); R["m6"] = dict(n=len(ts), start=str(ts.index[0]), end=str(ts.index[-1]), n_train=len(trn), n_test=len(tst), adf_stat=float(adf[0]), adf_p=float(adf[1]),
                                    mean=float(ts.mean()), sd=float(ts.std()))
dec = seasonal_decompose(ts, model="additive", period=24)
fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
for a, (n_, s_, c_) in zip(ax, [("Observed", dec.observed, "teal"), ("Trend", dec.trend, "orange"), ("Seasonal (24 h)", dec.seasonal, "pink"), ("Residual", dec.resid, "olive")]):
    a.plot(s_, color=PAL[c_]); a.set_ylabel(n_, fontsize=8)
save("m6_decomp")
R["m6"]["season_strength"] = float(max(0, 1 - np.nanvar(dec.resid) / np.nanvar(dec.seasonal + dec.resid)))
fc = {}; h = len(tst)
fc["Average (mean of train)"] = np.repeat(trn.mean(), h)
fc["Moving average (w=6)"] = np.repeat(trn.iloc[-6:].mean(), h)
fc["Seasonal naive (24 h)"] = trn.iloc[-24:].values[:h]
fc["Simple exp. smoothing"] = SimpleExpSmoothing(trn.values, initialization_method="estimated").fit().forecast(h)
t_ = np.arange(len(trn)); lr = np.polyfit(t_, trn.values, 1); fc["Linear trend regression"] = np.polyval(lr, np.arange(len(trn), len(trn) + h))
best_arima, best_aic = None, 1e18
for pdq in [(p, d_, q) for p in range(3) for d_ in range(2) for q in range(3)]:
    try:
        mm_ = ARIMA(trn.values, order=pdq).fit()
        if mm_.aic < best_aic: best_arima, best_aic, arm = pdq, mm_.aic, mm_
    except Exception: pass
fc[f"ARIMA{best_arima}"] = arm.forecast(h); R["m6"]["arima_order"] = list(best_arima); R["m6"]["arima_aic"] = float(best_aic)
mase_d = np.mean(np.abs(np.diff(trn.values))); met = []
for k, f_ in fc.items():
    e = tst.values - f_; met.append(dict(method=k, mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))), mape=float(np.mean(np.abs(e / tst.values)) * 100), mase=float(np.mean(np.abs(e)) / mase_d)))
R["m6"]["metrics"] = met
plt.figure(figsize=(10, 4)); plt.plot(trn.index, trn, color=PAL["plum"], label="train"); plt.plot(tst.index, tst, color="black", lw=2, label="actual")
for (k, f_), c in zip(fc.items(), [PAL["gold"], PAL["lime"], PAL["pink"], PAL["teal"], PAL["brown"], PAL["orange"]]): plt.plot(tst.index, f_, "--", color=c, label=k)
plt.legend(fontsize=7, ncol=2); plt.title("Hourly listings retrieved: 24-hour hold-out forecasts"); plt.ylabel("listings / hour"); save("m6_forecast")

json.dump(R, open("results.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("done")
