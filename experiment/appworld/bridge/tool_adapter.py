"""Tool conversion helpers for the AppWorld neutral bridge."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from appworld.collections.api_docs import ApiDocCollection

from experiment.common.neutral import ToolSpec


def _json_example_to_schema(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        return {"type": "array", "items": _json_example_to_schema(value[0])}
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "properties": {
                str(key): _json_example_to_schema(child)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            },
            "required": [str(key) for key in value.keys()],
        }
    return {"type": "string", "description": f"Observed example type: {type(value).__name__}"}


def _parameter_to_json_schema(parameter_info: Mapping[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    param_type = str(parameter_info.get("type", "") or "").strip()
    if param_type:
        schema["type"] = param_type

    description = str(parameter_info.get("description", "") or "").strip()
    constraints = list(parameter_info.get("constraints") or [])
    if constraints:
        description = "\n".join(part for part in [description, *map(str, constraints)] if part)
    if description:
        schema["description"] = description

    if "default" in parameter_info and parameter_info.get("default") is not None:
        schema["default"] = parameter_info.get("default")
    return schema


def _response_schemas_to_json_schema(response_schemas: Mapping[str, Any] | None) -> dict[str, Any]:
    response_schemas = response_schemas or {}
    success_schema = response_schemas.get("success")
    failure_schema = response_schemas.get("failure")
    result: dict[str, Any] = {"type": "object", "properties": {}}
    if success_schema is not None:
        result["properties"]["success"] = _json_example_to_schema(success_schema)
    if failure_schema is not None:
        result["properties"]["failure"] = _json_example_to_schema(failure_schema)
    if not result["properties"]:
        return {}
    return result


def api_doc_to_spec(app_name: str, api_name: str, api_doc: Mapping[str, Any]) -> ToolSpec:
    """Convert one AppWorld API document into a neutral tool specification."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_info in list(api_doc.get("parameters") or []):
        if not isinstance(parameter_info, Mapping):
            continue
        parameter_name = str(parameter_info.get("name", "") or "").strip()
        if not parameter_name:
            continue
        properties[parameter_name] = _parameter_to_json_schema(parameter_info)
        if bool(parameter_info.get("required", False)):
            required.append(parameter_name)

    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_json_schema["required"] = required

    metadata = {
        "app_name": app_name,
        "api_name": api_name,
        "path": str(api_doc.get("path", "") or ""),
        "method": str(api_doc.get("method", "") or ""),
        "mutates_state": str(api_doc.get("method", "") or "").upper() != "GET",
        "response_example_summary": json.dumps(
            dict(api_doc.get("response_schemas") or {}),
            ensure_ascii=False,
            sort_keys=True,
        )[:2000],
    }

    return ToolSpec(
        name=f"{app_name}__{api_name}",
        description=str(api_doc.get("description", "") or "").strip(),
        parameters_json_schema=parameters_json_schema,
        returns_json_schema=_response_schemas_to_json_schema(
            api_doc.get("response_schemas")
            if isinstance(api_doc.get("response_schemas"), Mapping)
            else None
        ),
        metadata=metadata,
    )


def tools_to_specs(api_docs: ApiDocCollection | Mapping[str, Mapping[str, Any]]) -> tuple[ToolSpec, ...]:
    """Convert AppWorld API docs into neutral tool specifications."""
    specs: list[ToolSpec] = []
    for app_name in sorted(api_docs.keys()):
        api_name_to_doc = api_docs[app_name]
        for api_name in sorted(api_name_to_doc.keys()):
            specs.append(api_doc_to_spec(app_name, api_name, api_name_to_doc[api_name]))
    return tuple(specs)
