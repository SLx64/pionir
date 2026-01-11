import pytest
from pionir.core.metadata import Metadata

def test_metadata_init():
    data = {"key1": "value1", "key2": 2}
    m = Metadata(data)
    assert m["key1"] == "value1"
    assert m["key2"] == 2
    assert len(m) == 2

def test_metadata_getattr_setattr():
    m = Metadata()
    m["new_key"] = "new_value"
    assert m["new_key"] == "new_value"
    assert "new_key" in m

def test_metadata_del():
    m = Metadata({"a": 1})
    del m["a"]
    assert "a" not in m
    assert len(m) == 0

def test_metadata_iter():
    m = Metadata({"a": 1, "b": 2})
    keys = set(m)
    assert keys == {"a", "b"}

def test_metadata_repr():
    m = Metadata({"a": 1})
    assert repr(m) == "{'a': 1}"

def test_metadata_keys_values_items():
    data = {"a": 1, "b": 2}
    m = Metadata(data)
    assert set(m.keys()) == {"a", "b"}
    assert set(m.values()) == {1, 2}
    assert set(m.items()) == {("a", 1), ("b", 2)}

def test_metadata_get():
    m = Metadata({"a": 1})
    assert m.get("a") == 1
    assert m.get("b") is None
    assert m.get("b", 3) == 3

def test_metadata_update():
    m = Metadata({"a": 1})
    m.update({"b": 2}, c=3)
    assert m["a"] == 1
    assert m["b"] == 2
    assert m["c"] == 3

def test_metadata_clear():
    m = Metadata({"a": 1})
    m.clear()
    assert len(m) == 0

def test_metadata_pop():
    m = Metadata({"a": 1, "b": 2})
    assert m.pop("a") == 1
    assert "a" not in m
    assert m.pop("c", 3) == 3

def test_metadata_popitem():
    m = Metadata({"a": 1})
    assert m.popitem() == ("a", 1)
    assert len(m) == 0
