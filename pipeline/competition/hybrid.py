"""Shared parametric and LightGBM logic for the competition simulator.

The marimo notebook remains the analysis surface, while this module owns the
implementation used by both the notebook and the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.calendar_events import FEATURE_COLUMNS, monthly_features
from pipeline.competition import load as competition_load


def build_panel(flights: pd.DataFrame, intl_train: pd.DataFrame) -> pd.DataFrame:
    """Build the same market-month training panel used in Part A of the notebook."""
    flight_months = competition_load.flights_monthly_agg(flights).rename(
        columns={"Departure Country Name": "market"}
    )
    flight_months["market"] = flight_months["market"].astype("string").str.upper().str.strip()
    flight_months["ym_str"] = flight_months["ym"].astype(str)
    flight_columns = [
        "ym_str",
        "market",
        "Total_Seats",
        "Total_PAX",
        "Total_P2P",
        "Total_Transfer",
        "Total_Transit",
        "routes",
        "airlines",
    ]

    train_months = (
        intl_train.groupby(["ym", "Nationality"], dropna=False)
        .agg(Guests=("Guests_n", "sum"), NewArr=("New Arrivals_n", "sum"))
        .reset_index()
    )
    train_months["market"] = train_months["Nationality"].astype("string").str.upper().str.strip()
    train_months["ym_str"] = train_months["ym"].astype(str)
    train_months["month"] = train_months["ym"].dt.month
    return train_months.merge(
        flight_months[flight_columns], on=["ym_str", "market"], how="inner"
    )


def build_ground_truth(intl_train: pd.DataFrame, intl_test: pd.DataFrame) -> pd.DataFrame:
    """Combine available hotel targets without using test rows to fit the model."""
    train = intl_train.groupby(["ym", "Nationality"], dropna=False).agg(
        Guests=("Guests_n", "sum"), NewArr=("New Arrivals_n", "sum")
    ).reset_index()
    train["market"] = train["Nationality"].astype("string").str.upper().str.strip()
    train["ym_str"] = train["ym"].astype(str)

    test = intl_test.assign(
        NewArr=pd.to_numeric(intl_test["New Arrivals"], errors="coerce")
    ).groupby(["ym", "Nationality"], dropna=False).agg(NewArr=("NewArr", "sum")).reset_index()
    test["market"] = test["Nationality"].astype("string").str.upper().str.strip()
    test["ym_str"] = test["ym"].astype(str)
    return pd.concat(
        [train[["ym_str", "market", "Guests", "NewArr"]], test[["ym_str", "market", "NewArr"]]],
        ignore_index=True,
    )


def fit_params(panel: pd.DataFrame, cap: float | None = 5.0) -> dict:
    """Fit volume-weighted market capture, monthly seasonality, and ALOS ratios."""
    grouped = panel.groupby("market").agg(
        New=("NewArr", "sum"), P2P=("Total_P2P", "sum"), G=("Guests", "sum")
    )
    capture = grouped["New"] / grouped["P2P"]
    if cap is not None:
        capture = capture.clip(upper=cap)

    temp = panel.merge(capture.rename("cap"), left_on="market", right_index=True)
    numerator = temp.groupby("month")["NewArr"].sum()
    denominator = (temp["Total_P2P"] * temp["cap"]).groupby(temp["month"]).sum()
    season = numerator / denominator
    season = season / season.mean()
    alos = grouped["G"] / grouped["New"]
    return {
        "capture": capture,
        "season": season,
        "alos": alos,
        "global_capture": panel["NewArr"].sum() / panel["Total_P2P"].sum(),
        "global_alos": panel["Guests"].sum() / panel["NewArr"].sum(),
    }


def simulate(
    panel: pd.DataFrame,
    params: dict,
    seats_mult: float = 1.0,
    lf_mult: float = 1.0,
    p2p_mult: float = 1.0,
    capture_mult: float = 1.0,
    alos_mult: float = 1.0,
    season_mult: float = 1.0,
    lf_target: float | None = None,
    capture_rate: float | None = None,
) -> pd.DataFrame:
    """Run the exact seats → PAX → P2P → arrivals → guest-nights chain."""
    out = panel.copy()
    out["Seats"] = out["Total_Seats"] * seats_mult
    measured_lf = out["Total_PAX"] / out["Total_Seats"].replace(0, np.nan)
    load_factor = (
        pd.Series(float(lf_target), index=out.index)
        if lf_target is not None
        else measured_lf * lf_mult
    )
    out["LoadFactor"] = load_factor
    out["PAX"] = out["Seats"] * out["LoadFactor"]
    p2p_share = out["Total_P2P"] / out["Total_PAX"].replace(0, np.nan)
    out["P2P"] = out["PAX"] * p2p_share * p2p_mult

    capture = out["market"].map(params["capture"]).fillna(params["global_capture"])
    if capture_rate is not None:
        capture = pd.Series(float(capture_rate), index=out.index)
    season = out["month"].map(params["season"]).fillna(1.0) * season_mult
    alos = out["market"].map(params["alos"]).fillna(params["global_alos"])
    out["CaptureUsed"] = capture * capture_mult
    out["ALOSUsed"] = alos * alos_mult
    out["NewArrivalHat"] = out["P2P"] * out["CaptureUsed"] * season
    out["GuestNightsHat"] = out["NewArrivalHat"] * out["ALOSUsed"]
    return out


def wmape(actual, pred) -> float:
    """Weighted mean absolute percentage error, the competition metric."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = np.abs(actual).sum()
    return float(100.0 * np.abs(actual - pred).sum() / denominator) if denominator else float("nan")


def build_featurizer(
    panel: pd.DataFrame,
    market_dtype: pd.CategoricalDtype | None = None,
    month_dtype: pd.CategoricalDtype | None = None,
):
    """Return Part B's leak-free features and category metadata."""
    market_dtype = market_dtype or pd.CategoricalDtype(sorted(panel["market"].dropna().unique()))
    month_dtype = month_dtype or pd.CategoricalDtype(list(range(1, 13)))

    def featurize(frame: pd.DataFrame, events: bool = True) -> pd.DataFrame:
        features = pd.DataFrame(index=frame.index)
        features["market"] = frame["market"].astype(market_dtype)
        features["month"] = frame["month"].astype(month_dtype)
        periods = pd.PeriodIndex(frame["ym_str"], freq="M")
        features["t"] = periods.year * 12 + periods.month - (2022 * 12 + 1)
        features["log_seats"] = np.log1p(frame["Total_Seats"])
        features["load_factor"] = frame["Total_PAX"] / frame["Total_Seats"]
        features["p2p_share"] = frame["Total_P2P"] / frame["Total_PAX"]
        features["transfer_share"] = (
            frame["Total_Transfer"] + frame["Total_Transit"]
        ) / frame["Total_PAX"]
        features["routes"] = frame["routes"].astype(float)
        features["airlines"] = frame["airlines"].astype(float)
        if events:
            event_features = monthly_features(periods)
            for column in FEATURE_COLUMNS:
                features[column] = event_features[column].to_numpy().astype(np.float32)
        return features

    categories = ["market", "month"]
    if len(panel) and featurize(panel.head(3), events=True).shape[1] != 13:
        raise AssertionError("expected 13 LightGBM features with calendar events")
    return categories, featurize


def make_lgbm_functions(featurize, categories, lgb=None) -> dict:
    """Construct the shared Part B LightGBM fitting and prediction functions."""
    if lgb is None:
        import lightgbm as lgb

    seeds = [0, 1, 2, 42, 2024]
    base = dict(
        objective="poisson",
        num_leaves=15,
        min_child_samples=20,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )

    def _dataset(frame, features, offset, reference=None, labels=None, objective="poisson"):
        target = frame["NewArr"].to_numpy(dtype=float) if labels is None else labels
        dataset = lgb.Dataset(
            features,
            label=target,
            init_score=offset if objective == "poisson" else None,
            categorical_feature=categories,
            reference=reference,
            free_raw_data=False,
        )
        if objective == "poisson":
            dataset.set_init_score(offset)
        return dataset

    def fit_poisson(train_iter, valid, max_rounds=2000, events=True):
        train_offset = np.log(train_iter["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        valid_offset = np.log(valid["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        dataset = _dataset(train_iter, featurize(train_iter, events), train_offset)
        valid_dataset = _dataset(
            valid, featurize(valid, events), valid_offset, reference=dataset
        )
        parameters = dict(base, seed=seeds[0])
        booster = lgb.train(
            parameters,
            dataset,
            num_boost_round=max_rounds,
            valid_sets=[valid_dataset],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return booster.best_iteration

    def fit_capture_models(train_final, best_iter, events=True):
        features = featurize(train_final, events)
        offset = np.log(train_final["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        dataset = _dataset(train_final, features, offset)
        models = []
        for seed in seeds:
            parameters = dict(base, seed=seed)
            models.append(lgb.train(parameters, dataset, num_boost_round=best_iter))
        return models

    def predict_capture_models(models, test, events=True):
        features = featurize(test, events)
        offset = np.log(test["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        predictions = [np.exp(model.predict(features, raw_score=True) + offset) for model in models]
        return np.mean(predictions, axis=0), predictions

    def predict_poisson(train_final, test, best_iter, events=True):
        models = fit_capture_models(train_final, best_iter, events)
        return predict_capture_models(models, test, events)

    def fit_alos_models(train, events=True):
        fit_rows = train[(train["NewArr"] > 0) & (train["Guests"] > 0)]
        features = featurize(fit_rows, events)
        target = np.log(
            fit_rows["Guests"].to_numpy(dtype=float) / fit_rows["NewArr"].to_numpy(dtype=float)
        )
        dataset = _dataset(fit_rows, features, None, labels=target, objective="regression")
        models = []
        for seed in seeds:
            parameters = dict(base, objective="regression", seed=seed)
            models.append(lgb.train(parameters, dataset, num_boost_round=300))
        return models

    def predict_alos_models(models, test, events=True):
        features = featurize(test, events)
        predictions = [np.exp(model.predict(features)) for model in models]
        return np.mean(predictions, axis=0), predictions

    def fit_alos(train, test):
        return predict_alos_models(fit_alos_models(train), test)[0]

    def train_model(frame, rounds, seed=0):
        features = featurize(frame)
        offset = np.log(frame["Total_P2P"].clip(lower=1).to_numpy(dtype=float))
        dataset = _dataset(frame, features, offset)
        return lgb.train(dict(base, seed=seed), dataset, num_boost_round=rounds)

    return {
        "fit_alos": fit_alos,
        "fit_alos_models": fit_alos_models,
        "predict_alos_models": predict_alos_models,
        "fit_poisson": fit_poisson,
        "fit_capture_models": fit_capture_models,
        "predict_capture_models": predict_capture_models,
        "predict_poisson": predict_poisson,
        "train_model": train_model,
    }


def fit_simulator_model(panel: pd.DataFrame) -> dict:
    """Train cached app models using Part B's forward validation protocol."""
    categories, featurize = build_featurizer(panel)
    tools = make_lgbm_functions(featurize, categories)
    periods = pd.PeriodIndex(panel["ym_str"], freq="M")
    train_window = panel.loc[periods <= pd.Period("2025-01", freq="M")]
    validation = panel.loc[periods > pd.Period("2025-01", freq="M")]
    if train_window.empty or validation.empty:
        months = sorted(pd.PeriodIndex(panel["ym_str"], freq="M").unique())
        if len(months) < 2:
            raise ValueError("At least two training months are required to fit LightGBM.")
        split = months[max(1, len(months) - 6)]
        train_window = panel.loc[pd.PeriodIndex(panel["ym_str"], freq="M") < split]
        validation = panel.loc[pd.PeriodIndex(panel["ym_str"], freq="M") >= split]
    best_iter = tools["fit_poisson"](train_window, validation)
    capture_models = tools["fit_capture_models"](panel, best_iter)
    alos_models = tools["fit_alos_models"](panel)
    return {
        "panel": panel,
        "best_iter": best_iter,
        "featurize": featurize,
        "capture_models": capture_models,
        "alos_models": alos_models,
        "tools": tools,
    }


def predict_simulator_model(bundle: dict, scenario: pd.DataFrame) -> pd.DataFrame:
    """Predict arrivals and guest-nights using the fitted five-seed ensembles."""
    tools = bundle["tools"]
    arrivals, arrival_seeds = tools["predict_capture_models"](
        bundle["capture_models"], scenario
    )
    alos, alos_seeds = tools["predict_alos_models"](bundle["alos_models"], scenario)
    result = scenario.copy()
    result["HybridNewArrivals"] = arrivals
    result["HybridALOS"] = alos
    result["HybridGuestNights"] = arrivals * alos
    for index, (arrival, stay) in enumerate(zip(arrival_seeds, alos_seeds)):
        result[f"HybridNewArrivalsSeed{index}"] = arrival
        result[f"HybridGuestNightsSeed{index}"] = arrival * stay
    return result


def save_simulator_artifact(
    bundle: dict,
    params: dict,
    raw_capture: pd.Series,
    artifact_dir: str | Path,
) -> Path:
    """Persist trained LightGBM boosters and compact chain calibration values."""
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_files = {"capture": [], "alos": []}
    for model_type, models in (
        ("capture", bundle["capture_models"]),
        ("alos", bundle["alos_models"]),
    ):
        for index, model in enumerate(models):
            filename = f"{model_type}_{index}.txt"
            model.save_model(str(destination / filename))
            model_files[model_type].append(filename)

    metadata = {
        "schema_version": 1,
        "best_iteration": int(bundle["best_iter"]),
        "markets": sorted(str(value) for value in bundle["panel"]["market"].dropna().unique()),
        "training_months": sorted(bundle["panel"]["ym_str"].astype(str).unique().tolist()),
        "chain_params": {
            "capture": {str(key): float(value) for key, value in params["capture"].items()},
            "raw_capture": {str(key): float(value) for key, value in raw_capture.items()},
            "season": {str(int(key)): float(value) for key, value in params["season"].items()},
            "alos": {str(key): float(value) for key, value in params["alos"].items()},
            "global_capture": float(params["global_capture"]),
            "global_alos": float(params["global_alos"]),
        },
        "model_files": model_files,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def load_simulator_artifact(artifact_dir: str | Path) -> tuple[dict, dict, pd.Series]:
    """Load the pre-trained ensemble and chain parameters without source data."""
    source = Path(artifact_dir)
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Pre-trained model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported simulator model artifact schema.")

    import lightgbm as lgb

    market_dtype = pd.CategoricalDtype(metadata["markets"])
    month_dtype = pd.CategoricalDtype(list(range(1, 13)))
    categories, featurize = build_featurizer(
        pd.DataFrame(), market_dtype=market_dtype, month_dtype=month_dtype
    )
    models = {
        name: [lgb.Booster(model_file=str(source / filename)) for filename in filenames]
        for name, filenames in metadata["model_files"].items()
    }
    if len(models.get("capture", [])) != 5 or len(models.get("alos", [])) != 5:
        raise ValueError("Expected five capture and five ALOS LightGBM boosters.")
    params_raw = metadata["chain_params"]
    params = {
        "capture": pd.Series(params_raw["capture"], dtype=float),
        "season": pd.Series({int(key): value for key, value in params_raw["season"].items()}, dtype=float),
        "alos": pd.Series(params_raw["alos"], dtype=float),
        "global_capture": float(params_raw["global_capture"]),
        "global_alos": float(params_raw["global_alos"]),
    }
    raw_capture = pd.Series(params_raw["raw_capture"], dtype=float)
    tools = make_lgbm_functions(featurize, categories, lgb)
    bundle = {
        "panel": None,
        "best_iter": int(metadata["best_iteration"]),
        "featurize": featurize,
        "capture_models": models["capture"],
        "alos_models": models["alos"],
        "tools": tools,
        "training_months": metadata.get("training_months", []),
    }
    return bundle, params, raw_capture
