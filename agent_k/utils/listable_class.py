import inspect
from functools import lru_cache


class ListableAttribute:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f"<{self.__class__.__name__}: name={self._name}, value={self._value}>"


class ListableClass:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check for duplicate class attribute values similar to Enum
        seen_values = {}
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not inspect.ismethod(value):
                if value in seen_values:
                    raise ValueError(
                        f"Duplicate value '{value}' found in class {cls.__name__} for '{key}' and '{seen_values[value]}'"
                    )
                seen_values[value] = key
                # Convert the class attribute value to ListableAttribute dynamically
                setattr(cls, key, ListableAttribute(key, value))

    @classmethod
    @lru_cache(maxsize=1)
    def list(cls):
        return [
            k for k, v in inspect.getmembers(cls) if isinstance(v, ListableAttribute)
        ]

    @classmethod
    @lru_cache(maxsize=1)
    def list_values(cls):
        return [
            v.value
            for k, v in inspect.getmembers(cls)
            if isinstance(v, ListableAttribute)
        ]

    @classmethod
    @lru_cache(maxsize=1)
    def info(cls):
        # Generate and return a nicely formatted string with attribute info
        attributes = [
            f"{k} = {str(v)}"
            for k, v in inspect.getmembers(cls)
            if isinstance(v, ListableAttribute)
        ]
        info_str = f"Class '{cls.__name__}' attributes:\n" + "\n".join(attributes)
        return info_str


"""
# Example usage:

class ColumnNames(ListableClass):
    COL1 = "col1"
    COL2 = "col2"
    # COL3 = "col1"  # This will trigger a ValueError due to duplicate values


class ColNames(ListableClass):
    APPLE_NEW_NAME = "apple"
    BANANA = "banana"
    CHERRY = "cherry"
    DATE = "date"


class ColNames2(ListableClass):
    APPLE = "apple"
    ORANGE = "orange"
    CHERRY = "cherry"


# Accessing attributes info
print(ColumnNames.info())

# Accessing attributes
print(ColumnNames.COL1.name)   # Output: 'COL1'
print(ColumnNames.COL1.value)  # Output: 'col1'

# Listing attributes
print(ColumnNames.list())  # Output: ['COL1', 'COL2']
print(ColumnNames.list_values())  # Output: ['col1', 'col2']

# Refactoring the class attribute names works as well
print(ColNames.APPLE_NEW_NAME.name)
print(ColNames.APPLE_NEW_NAME.value)
print(ColNames2.APPLE.name)
"""
