"""实验管理 REST API 路由

提供实验列表、详情、删除与对比端点，供 Web 界面与外部客户端调用。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from neurogenesis_lab.experiments import (
    ExperimentStore,
    config_diff_keys,
    flatten_config,
    remove_output_dir,
)

router = APIRouter(tags=["experiments"])

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_Templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


def _get_store(request: Request) -> ExperimentStore:
    """从 app.state 获取数据库路径并创建 ExperimentStore。"""
    return ExperimentStore(request.app.state.db_path)


# ── 页面端点 ────────────────────────────────────────────────────────────────


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request,
    filter: str | None = Query(None, description="过滤表达式"),
    limit: int = Query(50, ge=1, le=1000, description="最大返回条数"),
) -> HTMLResponse:
    """渲染实验列表页面。"""
    with _get_store(request) as store:
        records = store.list_experiments(filter_expression=filter, limit=limit)
    return _Templates.TemplateResponse(
        request,
        "experiments.html",
        {
            "experiments": [r.to_dict() for r in records],
            "filter": filter or "",
            "limit": limit,
        },
    )


@router.get("/experiments/compare", response_class=HTMLResponse)
async def experiments_compare_page(
    request: Request,
    ids: str = Query(..., description="逗号分隔的实验 ID"),
) -> HTMLResponse:
    """渲染实验对比页面。"""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    with _get_store(request) as store:
        records = [store.get(eid) for eid in id_list]
    # 用新变量名避开 mypy 对同名变量自引用推导式的收窄限制
    existing = [r for r in records if r is not None]
    flat_configs = [flatten_config(r.config) for r in existing]
    diff_keys = config_diff_keys(flat_configs)
    return _Templates.TemplateResponse(
        request,
        "experiment_detail.html",
        {
            "mode": "compare",
            "experiments": [r.to_dict() for r in existing],
            "diff_keys": diff_keys,
            "flat_configs": flat_configs,
        },
    )


@router.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail_page(
    request: Request,
    experiment_id: str,
) -> HTMLResponse:
    """渲染单个实验详情页面。"""
    with _get_store(request) as store:
        record = store.get(experiment_id)
    if record is None:
        return _Templates.TemplateResponse(
            request,
            "experiment_detail.html",
            {"mode": "detail", "experiment": None, "error": "实验不存在"},
            status_code=404,
        )
    return _Templates.TemplateResponse(
        request,
        "experiment_detail.html",
        {
            "mode": "detail",
            "experiment": record.to_dict(),
        },
    )


# ── JSON API 端点 ───────────────────────────────────────────────────────────


_VALID_SORT_FIELDS = {
    "created_at",
    "experiment_id",
    "status",
    "seed",
    "neurons",
    "ticks",
    "runtime_seconds",
}


@router.get("/api/experiments")
async def api_list_experiments(
    request: Request,
    filter: str | None = Query(None, description="过滤表达式"),
    limit: int = Query(50, ge=1, le=1000, description="最大返回条数"),
    offset: int = Query(0, ge=0, description="跳过的记录数"),
    sort_by: str = Query("created_at", description="排序字段"),
    sort_order: str = Query("desc", description="排序方向：asc 或 desc"),
) -> JSONResponse:
    """返回实验列表 JSON，支持分页与排序。"""
    # 校验排序字段
    if sort_by not in _VALID_SORT_FIELDS:
        sort_by = "created_at"
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    # 获取全部符合过滤条件的记录（不传 limit，由 API 层做分页）
    with _get_store(request) as store:
        records = store.list_experiments(filter_expression=filter, limit=0)

    total = len(records)

    # 排序
    reverse = sort_order == "desc"
    _numeric_fields = {"seed", "neurons", "ticks", "runtime_seconds"}

    def _sort_key(r: object) -> tuple:
        val = getattr(r, sort_by, None)
        if val is None:
            return (1, 0, "")
        if sort_by in _numeric_fields:
            return (0, val, "")
        return (0, 0, str(val))

    records.sort(key=_sort_key, reverse=reverse)
    # None 值始终排在末尾
    none_records = [r for r in records if getattr(r, sort_by, None) is None]
    non_none_records = [r for r in records if getattr(r, sort_by, None) is not None]
    records = non_none_records + none_records

    # 分页切片
    page = records[offset : offset + limit]

    return JSONResponse(
        {
            "experiments": [r.to_dict() for r in page],
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    )


@router.get("/api/experiments/compare")
async def api_compare_experiments(
    request: Request,
    ids: str = Query(..., description="逗号分隔的实验 ID"),
) -> JSONResponse:
    """返回多实验对比数据 JSON。"""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    with _get_store(request) as store:
        records = [store.get(eid) for eid in id_list]
    existing = [r for r in records if r is not None]
    flat_configs = [flatten_config(r.config) for r in existing]
    diff_keys = config_diff_keys(flat_configs)
    return JSONResponse(
        {
            "records": [r.to_dict() for r in existing],
            "diff_keys": diff_keys,
            "flat_configs": flat_configs,
        }
    )


@router.get("/api/experiments/{experiment_id}")
async def api_experiment_detail(
    request: Request,
    experiment_id: str,
) -> JSONResponse:
    """返回单个实验详情 JSON，不存在时返回 404。"""
    with _get_store(request) as store:
        record = store.get(experiment_id)
    if record is None:
        return JSONResponse({"detail": "实验不存在"}, status_code=404)
    return JSONResponse(record.to_dict())


@router.delete("/api/experiments/{experiment_id}")
async def api_delete_experiment(
    request: Request,
    experiment_id: str,
    keep_files: bool = Query(False, description="是否保留输出文件"),
) -> JSONResponse:
    """删除实验记录，可选是否同时删除输出目录。"""
    with _get_store(request) as store:
        record = store.get(experiment_id)
        if record is None:
            return JSONResponse({"detail": "实验不存在"}, status_code=404)
        if not keep_files:
            remove_output_dir(record.experiment_id, record.output_dir)
        store.delete(experiment_id)
    return JSONResponse({"status": "deleted", "id": experiment_id})


@router.get("/api/experiments/{experiment_id}/output")
async def api_experiment_output(
    request: Request,
    experiment_id: str,
    limit: int = Query(100, ge=1, le=10000, description="最大返回行数"),
) -> JSONResponse:
    """获取实验输出数据（run.jsonl）。

    返回前 N 条记录，并标注总行数与是否截断。
    """
    with _get_store(request) as store:
        record = store.get(experiment_id)
    if record is None:
        return JSONResponse({"detail": "实验不存在"}, status_code=404)

    output_file = record.output_file
    if not output_file or not Path(output_file).is_file():
        return JSONResponse({"lines": [], "total_lines": 0, "truncated": False})

    lines: list[dict] = []
    total_lines = 0
    with Path(output_file).open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total_lines += 1
            if len(lines) < limit:
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    lines.append({"_raw": raw_line})

    return JSONResponse(
        {
            "lines": lines,
            "total_lines": total_lines,
            "truncated": total_lines > limit,
        }
    )
