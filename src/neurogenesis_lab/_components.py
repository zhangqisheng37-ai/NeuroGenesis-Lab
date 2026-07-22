"""组件委托混入基类 — 将 dataclass 字段透明代理到子组件。

用法:
    @dataclass
    class MyClass(ComponentMixin):
        _component_specs: ClassVar = [
            ("_comp_a", ComponentA, _COMP_A_FIELDS),
            ("_comp_b", ComponentB, _COMP_B_FIELDS),
        ]
        # ... 字段声明 ...
        def __post_init__(self) -> None:
            super().__post_init__()
            # 自定义初始化逻辑

模块级清理:
    ComponentMixin.cleanup_dataclass_defaults(MyClass)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    pass  # 仅用于类型检查占位


class ComponentMixin:
    """将 dataclass 字段透明代理到子组件的混入基类。

    子类需定义 ``_component_specs`` 类变量，格式为:
        [(attr_name, ComponentClass, fields_frozenset), ...]
    """

    _component_specs: ClassVar[list[tuple[str, type, frozenset[str]]]] = []

    @classmethod
    def _field_to_component(cls) -> dict[str, str]:
        """构建 field_name -> component_attr_name 映射（带缓存）。"""
        cache_attr = "_field_to_component_cache"
        if getattr(cls, cache_attr, None) is not None:
            return cast("dict[str, str]", getattr(cls, cache_attr))
        mapping: dict[str, str] = {}
        for comp_attr, _comp_cls, fields in cls._component_specs:
            for fname in fields:
                mapping[fname] = comp_attr
        setattr(cls, cache_attr, mapping)
        return mapping

    def __post_init__(self) -> None:
        """从 __dict__ 中提取组件字段，构建子组件实例，并清理扁平字段。"""
        all_fields: set[str] = set()
        for comp_attr, comp_cls, fields in self._component_specs:
            kwargs = {fname: self.__dict__[fname] for fname in fields if fname in self.__dict__}
            object.__setattr__(self, comp_attr, comp_cls(**kwargs))
            all_fields.update(fields)
        for fname in all_fields:
            self.__dict__.pop(fname, None)

    def __getattr__(self, name: str) -> object:
        mapping = self._field_to_component()
        comp_attr = mapping.get(name)
        if comp_attr is not None:
            return getattr(getattr(self, comp_attr), name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        # 组件初始化前回退到 object.__setattr__
        if not self._components_initialized():
            object.__setattr__(self, name, value)
            return
        mapping = self._field_to_component()
        comp_attr = mapping.get(name)
        if comp_attr is not None:
            setattr(getattr(self, comp_attr), name, value)
        else:
            object.__setattr__(self, name, value)

    def _components_initialized(self) -> bool:
        """检查所有组件是否已完成初始化。"""
        for comp_attr, _comp_cls, _fields in self._component_specs:
            if comp_attr not in self.__dict__:
                return False
        return True

    def __setstate__(self, state: dict) -> None:
        """pickle 反序列化兼容 — 处理旧格式（扁平字段，无组件）。"""
        self.__dict__.update(state)
        if not self._components_initialized():
            all_fields: set[str] = set()
            for comp_attr, comp_cls, fields in self._component_specs:
                kwargs = {fname: self.__dict__[fname] for fname in fields if fname in self.__dict__}
                object.__setattr__(self, comp_attr, comp_cls(**kwargs))
                all_fields.update(fields)
            for fname in all_fields:
                self.__dict__.pop(fname, None)

    @classmethod
    def all_component_fields(cls) -> frozenset[str]:
        """返回所有组件字段的并集。"""
        result: set[str] = set()
        for _comp_attr, _comp_cls, fields in cls._component_specs:
            result.update(fields)
        return frozenset(result)

    @staticmethod
    def cleanup_dataclass_defaults(target_cls: type) -> None:
        """删除 dataclass 在类上生成的默认值属性，避免遮蔽 __getattr__ 路由。

        Args:
            target_cls: 继承 ComponentMixin 的 dataclass 类。
        """
        if not issubclass(target_cls, ComponentMixin):
            raise TypeError(f"{target_cls.__name__} 必须继承 ComponentMixin")
        all_fields = target_cls.all_component_fields()
        for fname in all_fields:
            if fname in target_cls.__dict__:
                delattr(target_cls, fname)
