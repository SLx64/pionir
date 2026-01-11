from collections import UserDict


class Metadata(UserDict):
    def __init__(self, data: dict | None = None, **kwargs):
        super().__init__(data, **kwargs)

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, item):
        self.data[key] = item

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __repr__(self) -> str:
        return repr(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def clear(self):
        self.data.clear()

    def pop(self, key: str, *args):
        return self.data.pop(key, *args)

    def popitem(self):
        return self.data.popitem()
