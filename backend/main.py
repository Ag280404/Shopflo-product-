from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sklearn.linear_model import LogisticRegression


DATABASE_URL = "sqlite:///./checkout.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class CheckoutEvent(Base):
    __tablename__ = "checkout_events"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    event_type = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    user_type = Column(String)
    city = Column(String)
    pincode = Column(String)
    device = Column(String)
    cart_value = Column(Float)
    shipping_eta_shown = Column(Boolean)
    payment_method = Column(String)
    gateway = Column(String)
    coupon_status = Column(String)
    modules_used = Column(String)


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime, index=True)
    user_type = Column(String)
    city = Column(String)
    pincode = Column(String)
    device = Column(String)
    cart_value = Column(Float)
    shipping_eta_shown = Column(Boolean)
    payment_method = Column(String)
    gateway = Column(String)
    coupon_status = Column(String)
    modules_used = Column(String)
    outcome = Column(String)
    stage_started = Column(Boolean)
    stage_shipping = Column(Boolean)
    stage_payment = Column(Boolean)
    stage_complete = Column(Boolean)
    drop_off_stage = Column(String)


Base.metadata.create_all(engine)

app = FastAPI(title="Drop-off Detective")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


USER_TYPES = ["new", "returning"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad"]
DEVICES = ["mobile", "desktop"]
PAYMENT_METHODS = ["COD", "UPI_INTENT", "UPI_COLLECT", "CARD"]
GATEWAYS = ["G1", "G2"]
COUPON_STATUS = ["none", "applied", "invalid"]
MODULES = ["upsell", "discount_ladder", "freebies"]
OUTCOMES = ["converted", "abandoned", "payment_failed"]


class SeedResponse(BaseModel):
    inserted_sessions: int
    inserted_events: int


class OverviewMetrics(BaseModel):
    date: str
    conversion_rate: float
    abandonment_rate: float
    payment_fail_rate: float
    prepaid_share: float
    aov: float


class SegmentMetric(BaseModel):
    segment: str
    value: str
    total_sessions: int
    abandonment_rate: float
    conversion_rate: float


class RootCauseNarrative(BaseModel):
    title: str
    evidence: List[str]
    estimated_impact: str
    recommended_fix: str
    experiment_plan: Dict[str, Any]


class RootCauseItem(BaseModel):
    id: str
    dimension: str
    segment: str
    abandonment_rate: float
    baseline_rate: float
    lift: float
    z_score: float
    sample_size: int
    narrative: RootCauseNarrative


class RootCauseResponse(BaseModel):
    causes: List[RootCauseItem]
    anomalies: List[str]
    model_features: List[Dict[str, Any]]


class RootCauseDetail(BaseModel):
    cause: RootCauseItem


def mock_llm(context: Dict[str, Any]) -> RootCauseNarrative:
    title = f"{context['dimension'].title()} friction in {context['segment']}"
    evidence = [
        f"Abandonment rate {context['abandonment_rate']:.1%} vs baseline {context['baseline_rate']:.1%}.",
        f"Lift of {context['lift']:.1%} with z-score {context['z_score']:.2f}.",
        f"Observed across {context['sample_size']} sessions.",
    ]
    estimated_impact = f"Potentially {math.ceil(context['lift'] * 100)}% higher abandonment in this segment."
    recommended_fix = (
        "Reduce checkout friction by adjusting module knobs: "
        "enable trust badges, simplify payment retries, and tailor offers for the segment."
    )
    experiment_plan = {
        "variants": [
            {"name": "control", "changes": []},
            {
                "name": "variant_a",
                "changes": [
                    "Show faster ETA messaging",
                    "Highlight alternate payment method",
                ],
            },
        ],
        "targeting_rules": {
            "segment": context["segment"],
            "dimension": context["dimension"],
        },
        "success_metrics": ["conversion_rate", "payment_success_rate"],
        "guardrails": ["aov", "refund_rate"],
    }
    return RootCauseNarrative(
        title=title,
        evidence=evidence,
        estimated_impact=estimated_impact,
        recommended_fix=recommended_fix,
        experiment_plan=experiment_plan,
    )


def generate_events(session_id: str, start_time: datetime, payload: Dict[str, Any]) -> Tuple[List[CheckoutEvent], str]:
    outcome = payload["outcome"]
    events = []
    base = payload.copy()
    base.pop("outcome")
    events.append(
        CheckoutEvent(
            session_id=session_id,
            event_type="checkout_started",
            timestamp=start_time,
            **base,
        )
    )
    shipping_time = start_time + timedelta(minutes=random.randint(1, 3))
    events.append(
        CheckoutEvent(
            session_id=session_id,
            event_type="shipping_shown",
            timestamp=shipping_time,
            **base,
        )
    )
    payment_time = shipping_time + timedelta(minutes=random.randint(1, 3))
    events.append(
        CheckoutEvent(
            session_id=session_id,
            event_type="payment_initiated",
            timestamp=payment_time,
            **base,
        )
    )
    if outcome == "converted":
        events.append(
            CheckoutEvent(
                session_id=session_id,
                event_type="payment_success",
                timestamp=payment_time + timedelta(minutes=1),
                **base,
            )
        )
    elif outcome == "payment_failed":
        events.append(
            CheckoutEvent(
                session_id=session_id,
                event_type="payment_failed",
                timestamp=payment_time + timedelta(minutes=1),
                **base,
            )
        )
    else:
        events.append(
            CheckoutEvent(
                session_id=session_id,
                event_type="checkout_abandoned",
                timestamp=payment_time + timedelta(minutes=1),
                **base,
            )
        )
    return events, outcome


def stitch_sessions(db: Session) -> int:
    events = db.execute(select(CheckoutEvent)).scalars().all()
    grouped: Dict[str, List[CheckoutEvent]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    db.execute(delete(SessionSummary))
    summaries = []
    for session_id, items in grouped.items():
        items.sort(key=lambda e: e.timestamp)
        base = items[0]
        event_types = {event.event_type for event in items}
        stage_started = "checkout_started" in event_types
        stage_shipping = "shipping_shown" in event_types
        stage_payment = "payment_initiated" in event_types
        stage_complete = "payment_success" in event_types
        if "payment_failed" in event_types:
            outcome = "payment_failed"
            drop_off_stage = "payment"
        elif "checkout_abandoned" in event_types:
            outcome = "abandoned"
            drop_off_stage = "payment" if stage_payment else "shipping"
        else:
            outcome = "converted"
            drop_off_stage = "none"
        summaries.append(
            SessionSummary(
                session_id=session_id,
                start_time=base.timestamp,
                user_type=base.user_type,
                city=base.city,
                pincode=base.pincode,
                device=base.device,
                cart_value=base.cart_value,
                shipping_eta_shown=base.shipping_eta_shown,
                payment_method=base.payment_method,
                gateway=base.gateway,
                coupon_status=base.coupon_status,
                modules_used=base.modules_used,
                outcome=outcome,
                stage_started=stage_started,
                stage_shipping=stage_shipping,
                stage_payment=stage_payment,
                stage_complete=stage_complete,
                drop_off_stage=drop_off_stage,
            )
        )
    db.add_all(summaries)
    db.commit()
    return len(summaries)


def seed_data(db: Session, sessions: int = 5000) -> SeedResponse:
    db.execute(delete(CheckoutEvent))
    db.execute(delete(SessionSummary))
    db.commit()
    events_inserted = 0
    now = datetime.utcnow()
    for index in range(sessions):
        session_id = f"sess_{index}_{random.randint(1000, 9999)}"
        start_time = now - timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
        modules_used = random.sample(MODULES, random.randint(0, len(MODULES)))
        payload = {
            "user_type": random.choice(USER_TYPES),
            "city": random.choice(CITIES),
            "pincode": str(random.randint(400001, 400050)),
            "device": random.choice(DEVICES),
            "cart_value": round(random.uniform(399, 4999), 2),
            "shipping_eta_shown": random.choice([True, False]),
            "payment_method": random.choice(PAYMENT_METHODS),
            "gateway": random.choice(GATEWAYS),
            "coupon_status": random.choices(COUPON_STATUS, weights=[0.6, 0.3, 0.1])[0],
            "modules_used": json.dumps(modules_used),
            "outcome": random.choices(OUTCOMES, weights=[0.55, 0.3, 0.15])[0],
        }
        event_list, _ = generate_events(session_id, start_time, payload)
        db.add_all(event_list)
        events_inserted += len(event_list)
    db.commit()
    sessions_inserted = stitch_sessions(db)
    return SeedResponse(inserted_sessions=sessions_inserted, inserted_events=events_inserted)


def ensure_seeded() -> None:
    with SessionLocal() as db:
        total = db.execute(select(SessionSummary)).scalars().first()
        if total is None:
            seed_data(db)


def bucket_cart_value(value: float) -> str:
    if value < 500:
        return "<500"
    if value < 1000:
        return "500-999"
    if value < 2000:
        return "1000-1999"
    if value < 3000:
        return "2000-2999"
    return "3000+"


def bucket_pincode(pincode: str) -> str:
    if len(pincode) < 2:
        return "unknown"
    return f"{pincode[:2]}xxx"


def calculate_overview(summaries: List[SessionSummary]) -> List[OverviewMetrics]:
    by_day: Dict[str, List[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        date_key = summary.start_time.date().isoformat()
        by_day[date_key].append(summary)
    metrics = []
    for date_key in sorted(by_day.keys()):
        items = by_day[date_key]
        total = len(items)
        converted = sum(1 for item in items if item.outcome == "converted")
        abandoned = sum(1 for item in items if item.outcome == "abandoned")
        payment_failed = sum(1 for item in items if item.outcome == "payment_failed")
        prepaid = sum(1 for item in items if item.payment_method != "COD")
        aov = sum(item.cart_value for item in items) / total
        metrics.append(
            OverviewMetrics(
                date=date_key,
                conversion_rate=converted / total,
                abandonment_rate=abandoned / total,
                payment_fail_rate=payment_failed / total,
                prepaid_share=prepaid / total,
                aov=aov,
            )
        )
    return metrics


def segment_breakdown(summaries: List[SessionSummary]) -> List[SegmentMetric]:
    segments: Dict[Tuple[str, str], List[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        segments[("user_type", summary.user_type)].append(summary)
        segments[("device", summary.device)].append(summary)
        segments[("payment_method", summary.payment_method)].append(summary)
        segments[("gateway", summary.gateway)].append(summary)
        segments[("cart_value", bucket_cart_value(summary.cart_value))].append(summary)
        segments[("pincode", bucket_pincode(summary.pincode))].append(summary)
    results = []
    for (segment, value), items in segments.items():
        total = len(items)
        abandoned = sum(1 for item in items if item.outcome == "abandoned")
        converted = sum(1 for item in items if item.outcome == "converted")
        results.append(
            SegmentMetric(
                segment=segment,
                value=value,
                total_sessions=total,
                abandonment_rate=abandoned / total,
                conversion_rate=converted / total,
            )
        )
    return results


def compute_root_causes(summaries: List[SessionSummary]) -> RootCauseResponse:
    total = len(summaries)
    baseline_abandon = sum(1 for item in summaries if item.outcome == "abandoned") / total
    candidate_segments: Dict[Tuple[str, str], List[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        candidate_segments[("device", summary.device)].append(summary)
        candidate_segments[("payment_method", summary.payment_method)].append(summary)
        candidate_segments[("gateway", summary.gateway)].append(summary)
        candidate_segments[("user_type", summary.user_type)].append(summary)
        candidate_segments[("coupon_status", summary.coupon_status)].append(summary)
        candidate_segments[("cart_value", bucket_cart_value(summary.cart_value))].append(summary)
    scored = []
    for (dimension, segment), items in candidate_segments.items():
        if len(items) < 50:
            continue
        abandoned = sum(1 for item in items if item.outcome == "abandoned")
        rate = abandoned / len(items)
        lift = (rate - baseline_abandon) / max(baseline_abandon, 1e-6)
        std = math.sqrt(baseline_abandon * (1 - baseline_abandon) / len(items))
        z_score = (rate - baseline_abandon) / std if std else 0.0
        if lift <= 0:
            continue
        scored.append(
            {
                "dimension": dimension,
                "segment": segment,
                "abandonment_rate": rate,
                "baseline_rate": baseline_abandon,
                "lift": lift,
                "z_score": z_score,
                "sample_size": len(items),
            }
        )
    scored.sort(key=lambda item: (item["z_score"], item["lift"]), reverse=True)
    top = scored[:5]
    causes = []
    for index, item in enumerate(top, start=1):
        narrative = mock_llm(item)
        causes.append(
            RootCauseItem(
                id=str(index),
                dimension=item["dimension"],
                segment=item["segment"],
                abandonment_rate=item["abandonment_rate"],
                baseline_rate=item["baseline_rate"],
                lift=item["lift"],
                z_score=item["z_score"],
                sample_size=item["sample_size"],
                narrative=narrative,
            )
        )
    anomalies = detect_anomalies(summaries)
    model_features = model_feature_importance(summaries)
    return RootCauseResponse(causes=causes, anomalies=anomalies, model_features=model_features)


def detect_anomalies(summaries: List[SessionSummary]) -> List[str]:
    by_day: Dict[str, List[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        by_day[summary.start_time.date().isoformat()].append(summary)
    if len(by_day) < 14:
        return ["Not enough history for anomaly detection."]
    dates = sorted(by_day.keys())
    last_week = dates[-7:]
    prev_week = dates[-14:-7]
    def abandonment_rate(days: List[str]) -> float:
        items = [item for day in days for item in by_day[day]]
        return sum(1 for item in items if item.outcome == "abandoned") / len(items)
    last_rate = abandonment_rate(last_week)
    prev_rate = abandonment_rate(prev_week)
    delta = last_rate - prev_rate
    if abs(delta) > 0.05:
        direction = "up" if delta > 0 else "down"
        return [f"Abandonment rate shifted {direction} by {delta:.1%} WoW."]
    return ["No major week-over-week anomalies detected."]


def model_feature_importance(summaries: List[SessionSummary]) -> List[Dict[str, Any]]:
    rows = []
    for summary in summaries:
        modules = json.loads(summary.modules_used or "[]")
        rows.append(
            {
                "user_type": summary.user_type,
                "device": summary.device,
                "payment_method": summary.payment_method,
                "gateway": summary.gateway,
                "coupon_status": summary.coupon_status,
                "shipping_eta_shown": summary.shipping_eta_shown,
                "cart_value_bucket": bucket_cart_value(summary.cart_value),
                "module_upsell": "upsell" in modules,
                "module_discount": "discount_ladder" in modules,
                "module_freebies": "freebies" in modules,
                "abandoned": summary.outcome == "abandoned",
            }
        )
    frame = pd.DataFrame(rows)
    y = frame.pop("abandoned").astype(int)
    X = pd.get_dummies(frame, drop_first=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    coefs = model.coef_[0]
    features = [
        {"feature": name, "importance": float(weight)}
        for name, weight in zip(X.columns, coefs)
    ]
    features.sort(key=lambda item: abs(item["importance"]), reverse=True)
    return features[:8]


@app.on_event("startup")
async def startup_event() -> None:
    ensure_seeded()


@app.post("/seed", response_model=SeedResponse)
async def seed_endpoint() -> SeedResponse:
    with SessionLocal() as db:
        return seed_data(db)


@app.get("/metrics/overview", response_model=List[OverviewMetrics])
async def metrics_overview() -> List[OverviewMetrics]:
    with SessionLocal() as db:
        summaries = db.execute(select(SessionSummary)).scalars().all()
    return calculate_overview(summaries)


@app.get("/metrics/segments", response_model=List[SegmentMetric])
async def metrics_segments() -> List[SegmentMetric]:
    with SessionLocal() as db:
        summaries = db.execute(select(SessionSummary)).scalars().all()
    return segment_breakdown(summaries)


@app.get("/root_causes", response_model=RootCauseResponse)
async def root_causes(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
    payment_method: Optional[str] = Query(None),
) -> RootCauseResponse:
    with SessionLocal() as db:
        summaries = db.execute(select(SessionSummary)).scalars().all()
    filtered = []
    for summary in summaries:
        if start_date and summary.start_time.date() < datetime.fromisoformat(start_date).date():
            continue
        if end_date and summary.start_time.date() > datetime.fromisoformat(end_date).date():
            continue
        if device and summary.device != device:
            continue
        if payment_method and summary.payment_method != payment_method:
            continue
        filtered.append(summary)
    if not filtered:
        return RootCauseResponse(causes=[], anomalies=["No data for filters."], model_features=[])
    return compute_root_causes(filtered)


@app.get("/root_causes/{cause_id}", response_model=RootCauseDetail)
async def root_cause_detail(cause_id: str) -> RootCauseDetail:
    with SessionLocal() as db:
        summaries = db.execute(select(SessionSummary)).scalars().all()
    response = compute_root_causes(summaries)
    if not response.causes:
        fallback = RootCauseItem(
            id="0",
            dimension="unknown",
            segment="unknown",
            abandonment_rate=0.0,
            baseline_rate=0.0,
            lift=0.0,
            z_score=0.0,
            sample_size=0,
            narrative=mock_llm(
                {
                    "dimension": "unknown",
                    "segment": "unknown",
                    "abandonment_rate": 0.0,
                    "baseline_rate": 0.0,
                    "lift": 0.0,
                    "z_score": 0.0,
                    "sample_size": 0,
                }
            ),
        )
        return RootCauseDetail(cause=fallback)
    for cause in response.causes:
        if cause.id == cause_id:
            return RootCauseDetail(cause=cause)
    return RootCauseDetail(cause=response.causes[0])
