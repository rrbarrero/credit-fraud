from dataclasses import asdict, dataclass, field


@dataclass
class EvaluationResult:
    model_name: str
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    accuracy: float = 0.0
    macro_avg: dict[str, float] = field(default_factory=dict)
    weighted_avg: dict[str, float] = field(default_factory=dict)
    pr_auc: float = 0.0

    @classmethod
    def from_report(
        cls, model_name: str, report_dict: dict[str, dict[str, float]], pr_auc: float
    ):

        return cls(
            model_name=model_name,
            per_class={
                k: v
                for k, v in report_dict.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            },
            accuracy=(
                report_dict["accuracy"]
                if isinstance(report_dict["accuracy"], float)
                else (
                    float(report_dict["accuracy"])
                    if isinstance(report_dict["accuracy"], (int, str))
                    else 0.0
                )
            ),
            macro_avg=report_dict["macro avg"],
            weighted_avg=report_dict["weighted avg"],
            pr_auc=pr_auc,
        )

    def to_dict(self) -> dict:
        return asdict(self)
