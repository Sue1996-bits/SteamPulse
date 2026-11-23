"""Read one frozen, compact demo export. Model coefficients use raw ordered features."""
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd


class Publication:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "feature_manifest.json").read_text())
        self.order = [f["name"] for f in self.meta["features"]]
        if not self.order or len(set(self.order)) != len(self.order) or set(self.order).intersection(
            [self.meta["label"], *self.meta["target_columns"]]
        ):
            raise ValueError("Invalid serving feature list")
        data = self.table("forecast_features")
        data = data[(data.feature_version == self.meta["feature_version"])
                    & (data.label_version == self.meta["label_version"])
                    & (data.source_snapshot == self.meta["snapshot_id"])]
        if data.empty or (pd.to_datetime(data.feature_max_date) > pd.to_datetime(data.as_of_date)).any():
            raise ValueError("Missing compatible features or future feature date")
        data["as_of_date"] = pd.to_datetime(data.as_of_date).dt.date
        data[self.order]  # check all manifest columns at startup
        self.features = data.set_index(["app_id", "as_of_date", "category"], verify_integrity=True)
        model_path = self.root / "model_manifest.json"
        self.model = json.loads(model_path.read_text()) if model_path.exists() else None
        if self.model:
            for name in ("feature_version", "label_version", "snapshot_id"):
                if self.model[name] != self.meta[name]:
                    raise ValueError("Incompatible model " + name)
            if self.model["features"] != self.order:
                raise ValueError("Model feature order mismatch")
            self.cutoff = date.fromisoformat(self.model["training_label_end"])
            self.spark_models = {}
            if self.model["format"] == "spark_pipeline":
                from pyspark.ml import PipelineModel
                from pipeline.config import get_spark

                self.spark = get_spark("steampulse-serving")
                for category in ("crash", "performance", "match_server", "monetization"):
                    path = self.root / self.model["categories"][category]["path"]
                    self.spark_models[category] = PipelineModel.load(str(path))
                return
            if self.model["format"] != "linear":
                raise ValueError("Unsupported model format")
            for category in ("crash", "performance", "match_server", "monetization"):
                model = self.model["categories"][category]
                if len(model["weights"]) != len(self.order) or not all(
                    math.isfinite(v) for v in [model["intercept"], *model["weights"]]
                ):
                    raise ValueError("Invalid linear model parameters")

    def table(self, name):
        df = pd.read_parquet(self.root / name)
        if "app_id" in df:
            df["app_id"] = df.app_id.astype(str)
        return df

    def prediction(self, row, category):
        if category in self.spark_models:
            from pyspark.sql.types import StructType
            from pyspark.sql.types import _parse_datatype_string

            schema = StructType()
            inputs = []
            for column in self.meta["features"]:
                dtype = column["type"]
                schema.add(column["name"], _parse_datatype_string(dtype), True)
                value = row[column["name"]]
                if pd.isna(value):
                    value = None
                elif dtype in {"int", "bigint", "smallint", "tinyint"}:
                    value = int(value)
                elif dtype in {"double", "float"}:
                    value = float(value)
                inputs.append(value)
            frame = self.spark.createDataFrame([tuple(inputs)], schema)
            prediction = self.spark_models[category].transform(frame).select("prediction").first()[0]
            if prediction is None or not math.isfinite(prediction):
                raise ValueError("Invalid model prediction")
            return min(1.0, max(0.0, float(prediction)))
        model = self.model["categories"][category]
        values = [float(row[col]) for col in self.order]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Model requires finite inputs")
        return min(1.0, max(0.0, model["intercept"] + sum(w*x for w, x in zip(model["weights"], values))))
