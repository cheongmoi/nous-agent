"""不依赖第三方库的 JSON Schema 子集校验器。"""

from __future__ import annotations

from typing import Any


class ToolArgumentError(ValueError):
    pass


def validate_arguments(schema: dict[str, Any], value: Any, path: str = "arguments") -> Any:
    """校验工具参数。

    Pi 上游通过 TypeBox 校验。这里实现工具声明最常用的 JSON Schema 子集，
    同时保留原对象，避免校验过程悄悄修改模型提交的参数。
    """

    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](value):
        raise ToolArgumentError(f"{path} 应为 {expected}，实际为 {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentError(f"{path} 必须是 {schema['enum']!r} 中的一个值")

    if expected == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolArgumentError(f"{path}.{key} 是必填参数")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ToolArgumentError(f"{path} 包含未知参数：{', '.join(sorted(extras))}")
        for key, child in properties.items():
            if key in value:
                validate_arguments(child, value[key], f"{path}.{key}")

    if expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            validate_arguments(schema["items"], item, f"{path}[{index}]")

    if expected in {"string", "array"}:
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolArgumentError(f"{path} 长度不能小于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolArgumentError(f"{path} 长度不能大于 {schema['maxLength']}")

    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolArgumentError(f"{path} 不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolArgumentError(f"{path} 不能大于 {schema['maximum']}")

    return value
