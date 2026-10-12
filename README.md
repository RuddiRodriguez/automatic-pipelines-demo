# Automatic Pipelines

Compare machine learning models by changing a settings file instead of rewriting the training code.

This public demonstration takes fictional survey responses, prepares numerical and categorical columns, compares three classifiers, and saves scores and trained models with MLflow. It demonstrates an experimentation workflow, not real customer behaviour or business performance.

## Run the demonstration

Use Python 3.11. Download the repository first:

```bash
git clone https://github.com/RuddiRodriguez/automatic-pipelines-demo.git
cd automatic-pipelines-demo
```

Then install and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/model_selection.py
```

The sample CSV is included. No account, API key, paid service, or private dataset is needed.

Open recorded experiments:

```bash
./.venv/bin/python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001
```

Visit http://127.0.0.1:5001. The explicit Python path uses this project’s installation even if Anaconda or another Python installation is active. Each experiment contains one parent run and three model runs. Open a model run to inspect scores, settings, a saved model, and an HTML diagram of its preparation steps.

The script also writes `results/comparison.csv` and one `results/*_pipeline.html` diagram per model. These generated files stay local.

## What happens

1. Read `config/config.yml` and load the CSV.
2. Apply configured row filters. Rows with missing outcomes are removed.
3. Reserve 20% of responses for testing, keeping similar proportions of both outcomes.
4. Fill missing numerical values with training medians and standardise their scale. Fill missing categories and convert categories into numerical columns.
5. Compare a most-common-outcome baseline, logistic regression, and random forest across five training splits.
6. Fit each full pipeline on the training portion and evaluate it on the reserved test portion.
7. Record settings, scores, models, and diagrams in MLflow.

Preparation is fitted inside each training split. Test records do not determine missing-value replacements or scaling. The saved model includes preparation, so new records follow the same steps.

## Synthetic data

`examples/synthetic_survey.csv` contains 1,000 fictional responses generated with seed 42. Inputs include age, monthly visits, satisfaction, value rating, and shopping channel. The two-group outcome is whether a fictional respondent would recommend a service.

The generator deliberately makes satisfaction, value rating, visits, and channel affect recommendation probability, then samples outcomes with randomness. Some ratings are missing to demonstrate missing-value handling. Age is an unrelated input. No source dataset, client records, or real survey observations were used.

Recreate the exact CSV:

```bash
python src/generate_demo_data.py
```

## Change an experiment

Edit `config/config.yml` to change preparation steps, model settings, or the dataset path. Relative data paths start at the repository root. Use `--config path/to/settings.yml` to choose another settings file.

Supported models in this demo: `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`. Preparation steps are listed in `STEPS` in `src/model_selection.py`. The parallel settings lists must have matching lengths; entries in the same position describe one step.

The reader also supports SPSS `.sav` files. Keep private files in ignored `data/`, change `data_kind` to describe your source, and use a new local configuration. This demonstration expects a two-group target encoded as 0 and 1.

## Read the results

- **Accuracy:** proportion of correct predictions.
- **F1:** balance between finding positive outcomes and avoiding false positives. The baseline can score zero because it never predicts a positive outcome.
- **ROC AUC:** how well scores rank positive outcomes above negative outcomes; 0.5 means chance-level ranking.
- **CV:** average across five training splits. **Test:** score on the reserved 20%.

Compare models using training-split scores. Test scores provide an illustrative final check; repeatedly choosing settings from those scores would make that check less independent. Synthetic results show the workflow functioning, not performance on real data.

See [the verified demo run](docs/demo-results.md) for recorded results.

## Project files

- `src/model_selection.py`: configuration, training, evaluation, and recording.
- `src/custom_transformers.py`: reusable data preparation components.
- `src/mlflow_helper.py`: local experiment recording.
- `src/generate_demo_data.py`: reproducible fictional data.
- `config/config.yml`: complete example experiment.
- `tests/test_demo.py`: repeatability, saved-model loading, and training-only preparation checks.

Earlier exploratory notebooks, generated files, unfinished form, and optional AI database-query experiment are excluded from this focused public demo. The working demonstration runs from the command line.

## Verify

```bash
python -m unittest discover -s tests -v
```

## Portfolio description

Built a configuration-driven machine learning workflow that prepares tabular data, compares classification models, and records experiments with MLflow. The public example uses reproducible synthetic survey data, keeps preparation within training splits, and saves complete prediction pipelines with their scores and settings.
