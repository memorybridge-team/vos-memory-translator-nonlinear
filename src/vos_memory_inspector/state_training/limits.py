"""비용 시계와 epoch 경계 예측. Torch 없이 운영 controller도 같은 계산을 쓴다."""
from datetime import datetime, timezone
import math
import time
from .common import require


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "UTC_TIMEZONE_REQUIRED")
    return parsed.timestamp()


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat().replace("+00:00", "Z")


def limits(config, now=None):
    now = time.time() if now is None else now
    start = timestamp(config["approval_start_utc"])
    rate, budget = config["pod_hourly_rate"], config["budget_usd"]
    require(all(math.isfinite(x) and x > 0 for x in (rate, budget, config["max_hours"])), "POSITIVE_FINITE_LIMITS_REQUIRED")
    require(start <= now + 5, "APPROVAL_START_IN_FUTURE")
    end = min(timestamp(config["deadline_utc"]), start + config["max_hours"] * 3600, start + budget / rate * 3600)
    reserve = config["artifact_reserve_seconds"]
    require(math.isfinite(reserve) and reserve >= 120, "ARTIFACT_RESERVE_MIN_120_SECONDS")
    used = max(0, now - start)
    return dict(approval_start_utc=utc(start), hard_deadline_utc=utc(end),
                training_deadline_utc=utc(end - reserve), remaining_seconds=max(0, end - now),
                elapsed_seconds=used, accrued_pod_usd=used * rate / 3600,
                budget_remaining_usd=max(0, budget - used * rate / 3600),
                artifact_reserve_seconds=reserve, pod_hourly_rate=rate,
                pod_stopped=False, billing_is_not_stopped_by_process_watchdog=True)


def epoch_forecast(history, max_epochs, now, end, rate=None):
    durations = [float(h["wall_seconds_including_checkpoint"]) for h in history if "wall_seconds_including_checkpoint" in h]
    require(durations and all(math.isfinite(x) and x > 0 for x in durations), "COMPLETE_EPOCH_TIMING_REQUIRED")
    remaining = max(0, max_epochs - history[-1]["epoch"])
    average = sum(durations) / len(durations)
    # 최근 지연 증가를 평균으로 희석하지 않고, 저장 후 다음 epoch 착수 여부를 판단한다.
    conservative = max(durations[-3:]) * 1.25 + 60
    return dict(measured_complete_epochs=len(durations), last_complete_epoch=history[-1]["epoch"],
                last_epoch_seconds=durations[-1], average_epoch_seconds=average,
                next_epoch_reserved_seconds=conservative,
                max_epochs_remaining=remaining, eta_average_seconds=remaining * average,
                eta_conservative_seconds=remaining * conservative,
                eta_average_pod_usd=remaining * average * rate / 3600 if rate else None,
                eta_conservative_pod_usd=remaining * conservative * rate / 3600 if rate else None,
                can_finish_max_epochs_before_training_deadline=now + remaining * conservative < end,
                can_start_next_epoch=now + conservative < end,
                forecast_is_not_a_completion_guarantee=True)
