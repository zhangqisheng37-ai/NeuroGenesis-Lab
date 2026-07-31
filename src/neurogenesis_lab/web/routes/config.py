"""配置管理 REST API 路由

提供配置 Schema 查询、校验、TOML 导出与预设模板管理功能。
"""

from __future__ import annotations

import dataclasses
from dataclasses import Field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from neurogenesis_lab.config import (
    EvolutionParams,
    NetworkParams,
    NeuronParams,
    SimConfig,
    sim_config_from_dict,
)

router = APIRouter(prefix="/api/config", tags=["config"])

# experiments/ 目录相对于项目根目录
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[4] / "experiments"


# ── Schema 生成辅助 ──────────────────────────────────────────


def _python_type_to_str(tp: Any) -> str:
    """将 Python 类型转换为字符串表示。"""
    if tp is int:
        return "int"
    if tp is float:
        return "float"
    if tp is bool:
        return "bool"
    if tp is str:
        return "str"
    return "any"


def _get_default(field: Field) -> Any:
    """获取 dataclass 字段的默认值。"""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None


def _generate_schema() -> dict[str, dict[str, dict[str, Any]]]:
    """从 dataclass 元信息生成配置 schema。

    遍历 SimConfig 及各子配置 dataclass 的字段，提取名称、类型与默认值。

    Returns:
        嵌套字典，顶层键为分组名（simulation/neuron/network/evolution），
        值为该分组下各字段的描述字典。
    """
    schema: dict[str, dict[str, dict[str, Any]]] = {}
    for cls_name, cls in [
        ("simulation", SimConfig),
        ("neuron", NeuronParams),
        ("network", NetworkParams),
        ("evolution", EvolutionParams),
    ]:
        fields_schema: dict[str, dict[str, Any]] = {}
        for f in dataclasses.fields(cls):
            if f.name == "schema_version":
                continue
            # 跳过组合型子配置字段（simulation 分组下的 neuron/network/evolution）
            if f.name in ("neuron", "network", "evolution"):
                continue
            field_info: dict[str, Any] = {
                "type": _python_type_to_str(f.type),
                "default": _get_default(f),
            }
            fields_schema[f.name] = field_info
        schema[cls_name] = fields_schema
    return schema


# ── TOML 序列化辅助 ──────────────────────────────────────────


def _toml_value(value: Any) -> str:
    """将 Python 值转换为 TOML 值字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return f"[{', '.join(_toml_value(v) for v in value)}]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
            + " }"
        )
    return str(value)


def _dict_to_toml(data: dict[str, Any], section: str = "") -> str:
    """将字典序列化为 TOML 格式字符串。

    先写标量值，再写子表（inline table 或 [section] 块）。

    Args:
        data: 待序列化的字典。
        section: 当前 TOML 节名前缀，顶层调用时为空字符串。

    Returns:
        TOML 格式字符串。
    """
    lines: list[str] = []
    # 先写标量值
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
    # 再写子表
    for key, value in data.items():
        if isinstance(value, dict):
            header = f"{section}.{key}" if section else key
            lines.append(f"\n[{header}]")
            for k, v in value.items():
                lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


# ── 配置构建辅助 ──────────────────────────────────────────────


def _build_sim_config(config_data: dict[str, Any]) -> SimConfig:
    """从 JSON 字典构建 SimConfig 实例（含字符串→数值的类型转换）。

    Args:
        config_data: 前端提交的配置字典，可包含 simulation/neuron/network/evolution 子键。

    Returns:
        构建完成的 SimConfig 实例。
    """
    return sim_config_from_dict(config_data)


# ── 路由端点 ──────────────────────────────────────────────────


@router.get("/schema")
async def get_schema() -> JSONResponse:
    """返回 SimConfig 的 JSON Schema。

    从 dataclass 元信息反射生成，包含各字段的类型与默认值。
    """
    schema = _generate_schema()
    return JSONResponse(schema)


@router.post("/validate")
async def validate_config(request: Request) -> JSONResponse:
    """验证配置字典的合法性。

    请求体为 JSON 配置字典，尝试构建 SimConfig 并调用 validate()。

    Returns:
        包含 valid（布尔）和 errors（错误列表）的 JSON 响应。
    """
    try:
        config_data: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"valid": False, "errors": ["请求体不是合法的 JSON"]})

    try:
        sim = _build_sim_config(config_data)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"valid": False, "errors": [f"参数构建失败: {exc}"]})

    try:
        errors = sim.validate()
    except (TypeError, ValueError) as exc:
        return JSONResponse({"valid": False, "errors": [f"参数类型错误: {exc}"]})
    return JSONResponse({"valid": len(errors) == 0, "errors": errors})


@router.post("/save")
async def save_config(request: Request) -> JSONResponse:
    """保存配置为 TOML 文件。

    请求体为 JSON，包含 name（文件名，不含扩展名）和配置数据。
    配置将经过校验后序列化并保存到 experiments/ 目录。

    Returns:
        包含保存路径的 JSON 响应。
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "请求体不是合法的 JSON"}, status_code=400)

    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "必须提供 name 字段"}, status_code=400)

    # 安全化文件名：仅保留字母数字、下划线和连字符
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    config_data = body.get("config", {})

    # 校验
    try:
        sim = _build_sim_config(config_data)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": f"参数构建失败: {exc}"}, status_code=400)

    errors = sim.validate()
    if errors:
        return JSONResponse({"error": "配置校验失败", "details": errors}, status_code=400)

    # 序列化为 TOML
    toml_data = sim.to_dict()
    # 移除 schema_version 以免写入 TOML
    toml_data.pop("schema_version", None)
    toml_str = _dict_to_toml(toml_data)

    # 保存文件
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _EXPERIMENTS_DIR / f"{safe_name}.toml"
    file_path.write_text(toml_str, encoding="utf-8")

    return JSONResponse({"path": f"experiments/{safe_name}.toml"})


@router.get("/presets")
async def list_presets() -> JSONResponse:
    """列出 experiments/ 目录下所有预设模板 TOML 文件。

    Returns:
        预设列表，每项包含 name 和 path。
    """
    if not _EXPERIMENTS_DIR.is_dir():
        return JSONResponse([])

    presets = []
    for toml_file in sorted(_EXPERIMENTS_DIR.glob("*.toml")):
        presets.append(
            {
                "name": toml_file.stem,
                "path": f"experiments/{toml_file.name}",
            }
        )
    return JSONResponse(presets)


@router.get("/presets/{name}")
async def load_preset(name: str) -> JSONResponse:
    """加载指定预设模板的内容。

    读取并解析 experiments/ 目录下指定的 TOML 文件，返回配置字典。

    Args:
        name: 预设文件名（不含 .toml 扩展名）。

    Returns:
        配置字典或错误信息。
    """
    # 安全化文件名
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    file_path = _EXPERIMENTS_DIR / f"{safe_name}.toml"

    if not file_path.is_file():
        return JSONResponse({"error": f"预设 '{safe_name}' 不存在"}, status_code=404)

    content = file_path.read_text(encoding="utf-8")
    # 简易 TOML 解析：按节拆分键值对
    parsed: dict[str, Any] = {}
    current_section: str | None = None

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            if current_section not in parsed:
                parsed[current_section] = {}
            continue
        if "=" in stripped:
            key, _, raw_value = stripped.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            value = _parse_toml_value(raw_value)
            if current_section and current_section in parsed:
                parsed[current_section][key] = value
            else:
                parsed[key] = value

    return JSONResponse(parsed)


def _parse_toml_value(raw: str) -> Any:
    """解析单个 TOML 值字符串为 Python 对象。

    支持布尔值、整数、浮点数、双引号字符串和行内列表。

    Args:
        raw: TOML 值的原始字符串。

    Returns:
        解析后的 Python 值。
    """
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_toml_value(v.strip()) for v in inner.split(",")]
    # 数值
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
