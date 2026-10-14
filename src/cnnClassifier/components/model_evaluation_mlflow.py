import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import numpy as np
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_metrics = {}

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Load without compiling to avoid Keras 2 -> Keras 3 reduction='auto' incompatibility
        model = tf.keras.models.load_model(path, compile=False)
        # Recompile with Keras 3 compatible settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(reduction="sum_over_batch_size"),
            metrics=["accuracy"]
        )
        return model
    

    def evaluation(self):
        # Support fallback to model/model.h5 if configured path does not exist
        model_path = Path(self.config.path_of_model)
        if not model_path.exists() and Path("model/model.h5").exists():
            model_path = Path("model/model.h5")

        self.model = self.load_model(model_path)
        self._valid_generator()

        # 1. Standard Loss and Accuracy from Keras
        self.score = self.model.evaluate(self.valid_generator)

        # 2. Detailed predictions for clinical classification metrics
        y_true = self.valid_generator.classes
        y_pred_probs = self.model.predict(self.valid_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Confusion Matrix components (0: Normal, 1: Tumor)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensitivity / Detection rate
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Optional ROC-AUC calculation
        roc_auc = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_pred_probs[:, 1]))
        except Exception:
            pass

        self.evaluation_metrics = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1_score, 4),
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            }
        }
        if roc_auc is not None:
            self.evaluation_metrics["roc_auc"] = round(roc_auc, 4)

        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.evaluation_metrics)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.evaluation_metrics.get("loss", float(self.score[0])),
                "accuracy": self.evaluation_metrics.get("accuracy", float(self.score[1])),
                "precision": self.evaluation_metrics.get("precision", 0.0),
                "recall": self.evaluation_metrics.get("recall", 0.0),
                "specificity": self.evaluation_metrics.get("specificity", 0.0),
                "f1_score": self.evaluation_metrics.get("f1_score", 0.0),
            })
            if "roc_auc" in self.evaluation_metrics:
                mlflow.log_metric("roc_auc", self.evaluation_metrics["roc_auc"])

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
