from typer.testing import CliRunner

from src.app import app


runner = CliRunner()


def test_eval_retrieval_exposes_json_report_option():
    result = runner.invoke(app, ["eval-retrieval", "--help"])

    assert result.exit_code == 0
    assert "--output-json" in result.output


def test_ask_does_not_expose_evaluation_report_option():
    result = runner.invoke(app, ["ask", "--help"])

    assert result.exit_code == 0
    assert "--output-json" not in result.output
