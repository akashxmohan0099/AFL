"""Metrics and experiment endpoints."""

from fastapi import APIRouter, Request

from api.services import metrics_service

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/experiments")
def get_experiments():
    return metrics_service.get_experiments()


@router.get("/calibration")
def get_calibration():
    return metrics_service.get_calibration_data()


@router.get("/backtest/{year}")
def get_backtest(year: int):
    return metrics_service.get_backtest_metrics(year)


@router.get("/multi-backtest/{year}")
def get_multi_backtest(year: int):
    return metrics_service.get_multi_backtest(year)


@router.get("/runtime")
def get_runtime_metrics(request: Request):
    return metrics_service.get_runtime_metrics(request.app)
