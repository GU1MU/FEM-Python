from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fem.application import ModelSession
from fem.application.runs import (
    B31_BEAM_FORMULATION,
    B31_RECOVERY_CONTRACT,
    B31_RESULT_POSITION,
)
from fem.core.model import AnalysisStep, FEMModel
from fem.io import load_result_archive, save_result_archive
from fem.post.stress import beam
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
)
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import make_solve_result_bundle
from tests.io.test_result_archive_v1 import _snapshot


def _solved_beam_session() -> tuple[ModelSession, str]:
    source = make_beam_field_characterization_result()
    model = FEMModel(
        mesh=deepcopy(source.model.mesh),
        name="phase6-beam-provenance",
        steps=(AnalysisStep("Load"),),
    )
    session = ModelSession()
    imported = session.prepare_import("phase6-beam.inp")
    session.accept_imported_model(imported.token, model)
    validation = session.prepare_validation("Load")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    solve = session.prepare_solve("Load", "B31 provenance")
    session.begin_run(solve.token)
    session.accept_run_succeeded(solve.token, make_solve_result_bundle(solve))
    return session, solve.run_id


def test_integration_point_stress_api_has_general_tensor_name() -> None:
    assert hasattr(beam, "recover_integration_point_stress")
    assert hasattr(beam, "BeamIntegrationPointStress")
    assert not hasattr(beam, "recover_integration_point_s11")
    assert not hasattr(beam, "BeamIntegrationPointS11")
    assert not hasattr(beam, "recover_section_stress")
    assert not hasattr(beam, "recover_section_end_stress")
    assert not hasattr(beam, "BeamSectionEndStress")


def test_new_beam_solve_and_archive_publish_b31_formulation_provenance(
    tmp_path: Path,
) -> None:
    session, run_id = _solved_beam_session()

    provenance = session.result_provenance_for(run_id)
    assert provenance is not None
    assert provenance.beam_formulation == B31_BEAM_FORMULATION
    assert provenance.beam_result_position == B31_RESULT_POSITION
    assert provenance.beam_recovery_contract == B31_RECOVERY_CONTRACT

    save = session.prepare_result_archive_save(run_id)
    assert save.archive.origin.provenance["beam_formulation"] == B31_BEAM_FORMULATION
    assert save.archive.origin.provenance["beam_result_position"] == B31_RESULT_POSITION
    assert (
        save.archive.origin.provenance["beam_recovery_contract"]
        == B31_RECOVERY_CONTRACT
    )

    path = tmp_path / "b31.femres"
    save_result_archive(path, save.archive)
    restored = ModelSession()
    restored.replace_from_result_archive(load_result_archive(path))
    restored_provenance = restored.current_result().provenance
    assert restored_provenance.beam_formulation == B31_BEAM_FORMULATION
    assert restored_provenance.beam_result_position == B31_RESULT_POSITION
    assert restored_provenance.beam_recovery_contract == B31_RECOVERY_CONTRACT


def test_legacy_archive_remains_unlabelled_by_current_b31_formulation() -> None:
    archive = _snapshot(make_beam_field_characterization_result, "legacy-beam")
    assert "beam_formulation" not in archive.origin.provenance

    session = ModelSession()
    session.replace_from_result_archive(archive, Path("legacy-beam.femres"))
    provenance = session.current_result().provenance

    assert provenance.beam_formulation is None
    assert provenance.beam_result_position is None
    assert provenance.beam_recovery_contract is None
    assert "beam_formulation" not in session.result_origin.provenance
