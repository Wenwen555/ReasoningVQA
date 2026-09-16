from dataclasses import dataclass

from .common import resolve_script


@dataclass(frozen=True)
class BackendAdapter:
    """Validated model-family operations exposed to the public pipeline."""

    name: str
    module: object

    @classmethod
    def from_module(cls, module):
        name = getattr(module, "NAME", None)
        if not isinstance(name, str) or not name:
            raise TypeError("backend module must declare a non-empty NAME")
        if not callable(getattr(module, "context", None)):
            raise TypeError(f"backend {name!r} must implement context(config)")
        operations = getattr(module, "OPERATIONS", None)
        if not isinstance(operations, dict):
            raise TypeError(f"backend {name!r} must declare OPERATIONS")
        for action, action_operations in operations.items():
            if action not in {"train", "evaluate"} or not isinstance(action_operations, dict):
                raise TypeError(f"backend {name!r} has invalid operation action {action!r}")
            for operation, script in action_operations.items():
                if not isinstance(operation, str) or not operation or not isinstance(script, str) or not script:
                    raise TypeError(f"backend {name!r} has an invalid {action} operation")
        return cls(name=name, module=module)

    def context(self, config):
        value = self.module.context(config)
        if not isinstance(value, dict):
            raise TypeError(f"backend {self.name!r} context must be a mapping")
        return value

    def resolve_operation(self, config, action, operation):
        if "grpo" in str(operation).lower():
            raise ValueError(f"GRPO is outside the current pipeline: {operation}")
        try:
            script = self.module.OPERATIONS[action][operation]
        except KeyError as exc:
            available = sorted(self.module.OPERATIONS.get(action, {}))
            raise ValueError(
                f"backend {self.name!r} has no {action} operation {operation!r}; available={available}"
            ) from exc
        return resolve_script(config, script)
