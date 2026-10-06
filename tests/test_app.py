from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_app_demo_and_settings_refresh():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].label == "Visible products"
    assert int(app.metric[0].value) > 0
    app.multiselect[0].set_value(["milk"]).run()
    assert not app.exception
    assert app.metric[1].value == "0"
    app.number_input[0].set_value(5).run()
    assert not app.exception
    assert app.metric[2].value == "1"


def test_empty_expected_selection_disables_scan():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    app.multiselect[0].set_value([]).run()
    assert app.button[0].disabled
    assert not app.exception
