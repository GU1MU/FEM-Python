from copy import copy, deepcopy
from dataclasses import replace
import pickle

import pytest

from fem import steps
from fem.core.model import AnalysisStep, OutputRequest, OutputSourceEvidence


def test_output_request_preserves_exact_variable_spelling_order_and_duplicates():
    request = OutputRequest(
        "FIELD",
        "NODE",
        (value for value in ("rf", "U", "rf", "CustomVariable")),
    )

    assert request.kind == "field"
    assert request.target == "node"
    assert request.variables == ("rf", "U", "rf", "CustomVariable")


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((1, "node", ()), "kind"),
        (("field", object(), ()), "target"),
        ((" ", "node", ()), "kind"),
        (("field", "\t", ()), "target"),
        (("field", "node", "U"), "variables"),
        (("field", "node", ("U", 1)), r"variables\[1\]"),
    ),
    ids=[
        "nonstring-kind", "nonstring-target", "blank-kind", "blank-target",
        "bare-variable", "nonstring-variable",
    ],
)
def test_output_request_rejects_nonstring_or_blank_labels_and_malformed_variables(
    arguments,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        OutputRequest(*arguments)


def test_output_metadata_is_a_detached_immutable_snapshot():
    thresholds = [0, 75, 100]
    nested = {"thresholds": thresholds}
    metadata = {
        "averaging": nested,
        "enabled": True,
        "note": None,
        "scale": 1.5,
    }

    request = OutputRequest("field", "element", ("S",), metadata)
    thresholds[1] = 80
    nested["late"] = "caller-owned"
    metadata["new"] = False

    assert request.metadata == {
        "averaging": {"thresholds": (0, 75, 100)},
        "enabled": True,
        "note": None,
        "scale": 1.5,
    }
    assert request.metadata["enabled"] is True
    for cloned in (copy(request), deepcopy(request), pickle.loads(pickle.dumps(request))):
        assert cloned == request
        with pytest.raises(TypeError):
            cloned.metadata["averaging"]["thresholds"][0] = 1

    with pytest.raises(TypeError):
        request.metadata["new"] = False
    with pytest.raises(TypeError):
        request.metadata["averaging"]["new"] = False
    with pytest.raises(TypeError):
        request.metadata["averaging"]["thresholds"][0] = 1


def test_replacing_output_variables_preserves_metadata_and_original_request() -> None:
    request = OutputRequest(
        "field",
        "node",
        ("U",),
        {"frequency": 1},
    )

    updated = replace(request, variables=("RF",))

    assert updated.metadata == {"frequency": 1}
    assert updated.variables == ("RF",)
    assert request.variables == ("U",)
    assert request.metadata == {"frequency": 1}
    with pytest.raises(TypeError):
        updated.metadata["frequency"] = 2


@pytest.mark.parametrize(
    "metadata",
    (
        {1: "non-string-key"},
        {"tuple": (1, 2)},
        {"custom": object()},
        {"nan": float("nan")},
        {"positive_infinity": float("inf")},
        {"negative_infinity": float("-inf")},
    ),
    ids=[
        "nonstring-key", "tuple", "custom-object", "nan",
        "positive-infinity", "negative-infinity",
    ],
)
def test_output_request_metadata_rejects_values_outside_strict_finite_json(
    metadata,
):
    with pytest.raises((TypeError, ValueError)):
        OutputRequest("field", "node", ("U",), metadata)


def test_output_request_metadata_rejects_cyclic_json_containers():
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict

    for metadata in ({"cycle": cyclic_list}, cyclic_dict):
        with pytest.raises(ValueError, match="cyclic"):
            OutputRequest("field", "node", ("U",), metadata)


def test_output_source_evidence_preserves_order_and_duplicates_after_source_mutation():
    parent_parameters = [["Frequency", "1"], ["Frequency", "2"]]
    parent_flags = ["FIELD", "FIELD"]
    child_parameters = [["NSET", "Tip"]]
    child_flags = ["FutureFlag"]

    evidence = OutputSourceEvidence(
        "ABAQUS",
        parent_parameters,
        parent_flags,
        child_parameters,
        child_flags,
    )
    request = OutputRequest(
        "field",
        "node",
        ("u", "u"),
        {"frequency": "1"},
        evidence,
    )
    parent_parameters[0][1] = "2"
    parent_flags.append("LATE")
    child_parameters.clear()
    child_flags.clear()

    assert evidence.source_kind == "abaqus"
    assert evidence.parent_parameters == (("Frequency", "1"), ("Frequency", "2"))
    assert evidence.parent_flags == ("FIELD", "FIELD")
    assert evidence.child_parameters == (("NSET", "Tip"),)
    assert evidence.child_flags == ("FutureFlag",)
    assert request.source_evidence == OutputSourceEvidence(
        "abaqus", (("Frequency", "1"), ("Frequency", "2")), ("FIELD", "FIELD"),
        (("NSET", "Tip"),), ("FutureFlag",),
    )
    assert deepcopy(request) == request
    with pytest.raises(TypeError):
        evidence.parent_parameters[0][1] = "changed"
    with pytest.raises(AttributeError):
        evidence.child_flags = ("changed",)

    with pytest.raises(TypeError, match="source_evidence"):
        OutputRequest("field", "node", ("U",), {}, object())


def test_output_helper_appends_in_order_and_snapshots_caller_data():
    step = AnalysisStep("load")
    variables = ["rf", "U", "rf"]
    thresholds = [25, 75]
    first = steps.output(
        step, "FIELD", "NODE", variables,
        averaging={"thresholds": thresholds},
    )
    second = steps.output(step, "history", "element", ["S"])
    variables.clear()
    thresholds[0] = 99

    assert step.outputs == (
        OutputRequest(
            "field", "node", ("rf", "U", "rf"),
            {"averaging": {"thresholds": [25, 75]}},
        ),
        OutputRequest("history", "element", ("S",)),
    )
    assert first == step.outputs[0]
    assert second == step.outputs[1]
