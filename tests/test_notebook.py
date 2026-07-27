from tldm.notebook import tldm as tldm_notebook


def test_notebook_disabled_description():
    """Test that set_description works for disabled tldm_notebook"""
    with tldm_notebook(1, disable=True) as t:
        t.set_description("description")


def test_notebook_disabled_display():
    """Test that display is a no-op for disabled tldm_notebook."""
    with tldm_notebook(1, disable=True) as t:
        assert t.display() is False
