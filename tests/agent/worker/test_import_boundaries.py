import subprocess
import sys


def test_worker_boundary_modules_do_not_load_execution_dependencies():
    source = """
import importlib
import sys

importlib.import_module("fem_agent.worker_protocol")
importlib.import_module("fem_agent.worker_integrity")
execution_modules = {
    "fem_agent.worker",
    "fem_agent.tools.exports",
    "fem_agent.tools.importing",
    "fem_agent.tools.registry",
    "fem_agent.tools.results",
    "fem_agent.tools.solving",
    "fem_agent.tools.validation",
}
loaded = execution_modules.intersection(sys.modules)
assert not loaded, sorted(loaded)
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_worker_public_imports_preserve_protocol_and_integrity_objects():
    import fem_agent.worker as worker_module
    import fem_agent.worker_integrity as integrity_module
    import fem_agent.worker_protocol as protocol_module

    for name in (
        "INSPECTION_SCHEMA_VERSION",
        "RESULT_QUERY_SCHEMA_VERSION",
        "WORKER_SCHEMA_VERSION",
        "InspectionRequest",
        "InspectionResponse",
        "InspectionWorkerError",
        "ResultQueryRequest",
        "ResultQueryWorkerError",
        "WorkerRequest",
        "WorkerResponse",
    ):
        assert getattr(worker_module, name) is getattr(protocol_module, name)
    for name in (
        "WorkerResponseIntegrityError",
        "load_verified_worker_response",
    ):
        assert getattr(worker_module, name) is getattr(integrity_module, name)
