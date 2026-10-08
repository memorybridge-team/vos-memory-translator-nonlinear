"""Float64 Welford/Chan 충분통계. 전체 tensor 보관과 batch R² 평균을 금지한다."""
import math
from .common import require

SPEC = dict(schema_version="cmmt.validation_state_r2.v1", target="paired Base+ state",
            spatial_axes="channel; observations=valid records,height,width", pointer_axes="coordinate; observations=valid records",
            variance_threshold=1e-12, variance_definition="population M2/count",
            constant_rule="exclude variance<=threshold dimensions from SSE/SST; report their MSE; all excluded => undefined",
            branch_weighting="1 - sum(SSE[eligible]) / sum(M2[eligible])",
            score_weighting="equal mean of MOSEv2 spatial,pointer and LVOSv2 spatial,pointer",
            precision="float64 Welford/Chan", negative_r2="preserve", tie_tolerance=1e-12,
            tie_policy="lower completed epoch", target_statistics="immutable validation index binding; never input normalization")
DATASETS = ("MOSEv2", "LVOSv2")
BRANCHES = ("spatial", "pointer")


class Moments:
    def __init__(self, dimensions):
        self.count = 0
        self.mean = [0.0] * dimensions
        self.m2 = [0.0] * dimensions

    def merge(self, count, mean, m2):
        require(count >= 0 and len(mean) == len(m2) == len(self.mean), "MOMENTS_SHAPE")
        require(all(math.isfinite(x) for x in mean + m2) and all(x >= 0 for x in m2), "NONFINITE_TARGET_STATISTICS")
        if not count:
            return
        total = self.count + count
        for i in range(len(mean)):
            delta = mean[i] - self.mean[i]
            self.m2[i] += m2[i] + delta * delta * self.count * count / total
            self.mean[i] += delta * count / total
        self.count = total

    def add(self, rows):
        # 작은 numeric fixture/일반 streaming API. tensor adapter는 batch Chan merge 사용.
        for row in rows:
            self.merge(1, list(map(float, row)), [0.0] * len(row))

    def payload(self):
        return dict(count=self.count, mean=self.mean, m2=self.m2)

    @classmethod
    def from_payload(cls, value):
        result = cls(len(value["mean"]))
        result.merge(**value)
        return result


def summarize(target_stats, sse, spec=SPEC):
    n, m2 = target_stats["count"], target_stats["m2"]
    require(n > 0 and len(m2) == len(sse) and len(sse) > 0, "EMPTY_BRANCH")
    require(all(math.isfinite(x) and x >= 0 for x in sse + m2), "NONFINITE_METRIC")
    eligible = [i for i, v in enumerate(m2) if v / n > spec["variance_threshold"]]
    constant = [i for i in range(len(m2)) if i not in eligible]
    sst = math.fsum(m2[i] for i in eligible)
    error = math.fsum(sse[i] for i in eligible)
    return dict(r2=1 - error / sst if eligible else None, sse=error, sst=sst,
                mse=math.fsum(sse) / (n * len(sse)), observations_per_dimension=n,
                eligible_dimensions=len(eligible), excluded_dimensions=len(constant),
                excluded_dimension_mse=math.fsum(sse[i] for i in constant) / (n * len(constant)) if constant else None)


def score(metrics):
    require(set(metrics) == set(DATASETS), "REQUIRED_DATASET_MISSING")
    values = []
    for dataset in DATASETS:
        require(set(metrics[dataset]) == set(BRANCHES), "REQUIRED_BRANCH_MISSING")
        for branch in BRANCHES:
            value = metrics[dataset][branch]["r2"]
            require(value is not None and math.isfinite(value), "UNDEFINED_OR_NONFINITE_R2")
            values.append(value)
    return math.fsum(values) / 4


class Coverage:
    def __init__(self, expected):
        self.expected = {tuple(x) for x in expected}
        require(len(self.expected) == len(expected), "DUPLICATE_EXPECTED_RECORD")
        self.seen = set()

    def add(self, refs):
        for ref in map(tuple, refs):
            require(ref in self.expected and ref not in self.seen, "FOREIGN_OR_DUPLICATE_RECORD")
            self.seen.add(ref)

    def finish(self):
        require(self.seen == self.expected, "MISSING_TAIL_RECORD")
        return len(self.seen)
