# Accident Risk Assessor

A team project for CS 4641 (Machine Learning) at Georgia Tech, spring 2023. Five students tried to predict whether a
crash happens, given a place, a time, and road, weather, driver and vehicle details. The data is the public crash
reports of Montgomery County, Maryland.

## The result

**On the fair test, the random forest scored 67.7% accuracy, below the 75.0% of always answering "no crash".** It was
trained on 155,707 real crash records plus three times as many "random negatives": real crash records with the
location, the time of day or the date swapped for another record's. Three of every four test rows are negatives, so a
model that always says "no crash" scores 75.0%. The forest found 651 of the 31,141 real crashes (2%). No model in the
project did better than that baseline on random negatives.

**The 98% figure comes from a different set of negatives.** The same random forest scored 98.2% when the negatives
were built by hand: each feature drawn from a probability distribution meant to match the real data, with a few
linked pairs such as vehicle make and model. Those rows are easy to tell apart from real records: 81.5% of them lie outside a latitude/longitude box that holds 96.2% of the real crashes. Location
alone gives most of them away. So the 98% shows how well the model spots generated rows, not how well it predicts
crashes.

The report flags both problems in its own words: the random-negative model "almost never correctly guesses
accidents" (p. 3), and a model on the hand-built set "may be overfitting to the dataset due to how the non-accident
samples were created" (p. 5).

## Results

Each model trains on 80% of a dataset and is scored on the other 20% (the neural network: 67% and 33%). "Crashes found"
is recall on the crash class. Report page numbers refer to `CS-4641-Project-Report.pdf`.

**Random negatives** (`data/balanced_data.csv`, 155,707 crashes and 467,121 negatives):

| Model | Accuracy | Crashes found | Source |
|---|---:|---:|---|
| Always answer "no crash" | 75.0% | 0% | 93,425 of the 124,566 test rows are negatives (support column of the random-forest output) |
| Random forest, 100 trees | 67.7% | 2% | `modelling/Metrics for Random Forest model with balanced_data.csv.png`; report p. 3 |
| XGBoost, all features | 71.6% | 17% (precision 40%) | report p. 4 (no screenshot in the repo) |
| XGBoost, 5 PCA components | 74.9% | 0% | `modelling/Metrics for XGBoost model with balanced_data.csv.png`; report p. 4 |
| LightGBM | 75.0% | 0% | `modelling/Metrics for light GBM model with balanced_data.csv.png` |
| Neural network, 27 → 1 | 74.9% | not reported | `modelling/ml-project-nn.png`; report p. 5 |
| Neural network on PCA data | about 50% | not reported | report p. 5 |

**Hand-built negatives** (`data/balanced_nonrandom_data.csv`, 155,707 crashes and 477,124 negatives):

| Model | Accuracy | Crashes found | Source |
|---|---:|---:|---|
| Random forest, 100 trees | 98.2% | 95% | `modelling/Metrics for Random Forest model with balanced_nonrandom_data.csv.png`; report p. 4 |
| LightGBM | 98.1% | 94% | `modelling/Metrics for light GBM model with balanced_nonrandom_data.csv.png` |
| XGBoost | 86.6% | 56% | `modelling/Metrics for XGBoost model with balanced_nonrandom_data.csv.png`; report p. 5 |
| Neural network, 27 → 1 | 75.5% | not reported | report p. 5 |

Notes on reading these numbers:

- Three of every four rows are negatives, so accuracy near 75% can mean a model learned nothing. On random negatives,
  LightGBM answered "no crash" for every test row and XGBoost on PCA data for all but 10 (their confusion matrices show
  it). The neural network's 74.9% and 75.5% are close to the share of negatives in its two datasets (75.0% and 75.4%).
- `modelling/gbdt.py` as committed trains XGBoost on 5 PCA components. The report does not say whether the 86.6% run
  used them.
- The 81.5% and 96.2% figures above come from the two committed CSVs: the box spans the 1st to 99th percentile of the
  real crashes' latitude and longitude (38.965–39.275° N, 76.936–77.315° W). 3.8% of real crashes and 3.8% of random
  negatives lie outside it.

## The data

- **Source:** Montgomery County's [Crash Reporting – Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data).
  The report and one file name call the area "Baltimore". The records are Montgomery County's, next to Washington,
  D.C. Their median point is 39.08° N, 77.11° W. The location filter in `data/Scripts/csv_cleaner.ipynb` keeps
  38.8–39.5° N and 76.8–77.7° W, which leaves Baltimore city outside.
- **Size:** `data/Prev Cleaning Stages/Original_Data.csv` holds 155,811 rows and 43 columns. After the location filter
  and cleaning, 155,707 crash records remain (the report's introduction says 155,108).
- **Features:** columns that describe the crash after it happened (collision type, severity, driver at fault and so on)
  were dropped. 24 columns remain in `data/Accident Data/Column_Cleaned_Data.csv`. The date and time became day of
  week, day of year, week of year, month, year and minute of the day.
- **PCA:** `data/Scripts/pca.py` keeps the fewest components that hold 99% of the variance. On the crash records that
  is 5 (`data/Accident Data/Accident_PCA_Data.csv`). It centres the columns but does not scale them, so columns with
  large ranges dominate the components.

Pipeline, as committed (the notebook and `pca.py` write `output.csv`; the stage files were renamed by hand):

```
Original_Data.csv (155,811 rows) ──csv_cleaner.ipynb: location filter──► Baltimore_Area.csv (155,707 rows)
  ──csv_cleaner.ipynb: column clean-up──► Cleaned_Data.csv ──► Accident Data/Column_Cleaned_Data.csv (24 columns)
Numerical_Data.csv (categories as integer codes; the script that made it is not in the repo)
Numerical_Data + Column_Cleaned_Data ──combination_numerical_and_text.py──► normalized_numerical_data.csv (crashes)
normalized_numerical_data ──nonaccidentscript.py─────────► data/balanced_data.csv (random negatives, 3 per crash)
                          ──(generator not in the repo)──► data/balanced_nonrandom_data.csv (hand-built negatives)
                          ──pca.py───────────────────────► Accident_PCA_Data.csv (5 components)
data/Balanced_Data_PCA.csv: 5 PCA components of balanced_data.csv (data/New_PCA_Data.csv is an identical copy)
```

## How to run

The code was written in 2023 for Python 3 with TensorFlow 2. Install the packages:

```bash
pip install "pandas<2" numpy scikit-learn xgboost lightgbm tensorflow matplotlib
```

`combination_numerical_and_text.py` calls `Series.dt.weekofyear`, which pandas deprecated in 1.1, so use pandas 1.x.

Models (run from `modelling/`; each prints a confusion matrix and accuracy):

```bash
cd modelling
python randomforest.py     # reads ../data/balanced_nonrandom_data.csv; change it to balanced_data.csv for random negatives
python lgbm.py             # same input file as randomforest.py
python gbdt.py             # XGBoost on 5 PCA components, hand-built negatives
python neural-net.py       # random negatives, 3 epochs
python location-dbscan.py  # DBSCAN (eps 0.03, min 1000 points) on a 30% sample of crash locations; slow
```

With scikit-learn 1.2.2 and pandas 1.5.3, `randomforest.py` prints the same confusion matrices and accuracies as the
committed screenshots: 0.9823650714641258 as committed, and 0.6772795144742546 with `balanced_data.csv`.

Data scripts: `data/Scripts/pca.py` runs from the repo root. `combination_numerical_and_text.py` and
`nonaccidentscript.py` read and write bare file names, so run them inside `data/Accident Data/` (the committed
`balanced_data.csv` lives one level up, in `data/`). `nonaccidentscript.py` is slow: it checks each new row against
every crash record. The plotting scripts in `visualizations/` and `csv_cleaner.ipynb` read the raw data.gov download
or paths on the authors' machines. Set `FILE_PATH` (or `file_path`) before running them.

## Who built what

Five students built this project. The contribution table on page 8 of the report names each member and their part.

- **Saahil Sanganeriya** built the random negatives with a teammate (`data/Scripts/nonaccidentscript.py`,
  `data/balanced_data.csv`), trained the random forest (`modelling/randomforest.py`) and ran the LightGBM comparison
  (`modelling/lgbm.py`). He also wrote the introduction and problem definition and took the data-cleaning tasks in the
  project timeline.
- **The other four members** cleaned and formatted the data, built the hand-built negatives, ran PCA, trained XGBoost
  and the neural network, ran DBSCAN, made the project page (`index.html`, with `_config.yml`, `layouts/` and
  `assets/`) and the presentation video, and wrote the final report.

Every commit in this repo is Saahil's because he uploaded the team's work here in September 2023.

Presentation video (uploaded by a teammate): <https://www.youtube.com/watch?v=FM7b4zGDyb0>

## Layout

```
CS-4641-Project-Report.pdf   final report (printed from the team's project page)
index.html, _config.yml,     a copy of the team's project page (a GitHub Pages site)
  layouts/, assets/
data/                        CSVs at each stage (see the pipeline above) and data/Scripts/
modelling/                   randomforest.py, gbdt.py (XGBoost), lgbm.py, neural-net.py, location-dbscan.py,
                             the metric screenshots, and test-neural-net.py (a Keras check on an unrelated public dataset)
visualizations/              frequency and location plots of the raw data, and the scripts that drew them
images/                      the team's timeline and contribution tables
```

## References

1. World Health Organisation. Global status report on road safety 2018; 17th June 2018; Available at https://www.who.int/publications/i/item/9789241565684
2. Centers for Disease Control and Prevention. WISQARS (Web-based Injury Statistics Query and Reporting System) [online]; 2020. Available at https://www.cdc.gov/injury/wisqars/index.html
3. Daniel Wilson. Using Machine Learning to Predict Car Accident Risk; 3rd May 2018; Available at https://medium.com/geoai/using-machine-learning-to-predict-car-accident-risk-4d92c91a7d57
4. Cigdem, A., and Cevher Ozden. "Predicting the severity of motor vehicle accident injuries in Adana-Turkey using machine learning methods and detailed meteorological data." Int. J. Intell. Syst. Appl. Eng 6.1 (2018): 72-79.
5. Al-Mistarehi, B. W., et al. "Using machine learning models to forecast severity level of traffic crashes by r studio and arcgis. front." Machine Learning Applications in Civil Engineering 16648714 (2022): 31.
