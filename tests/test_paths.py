from pathlib import Path

from sparmoe_vl.paths import repository_root, workspace_root


def test_repository_root_discovers_checkout(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "renamed-project"
    (checkout / "src" / "sparmoe_vl").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='sparmoe-vl'\n")
    nested = checkout / "experiments" / "example"
    nested.mkdir(parents=True)

    monkeypatch.delenv("SPARMOE_VL_ROOT", raising=False)
    monkeypatch.chdir(nested)
    assert repository_root() == checkout.resolve()


def test_environment_paths_override_discovery(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    workspace = tmp_path / "external-assets"
    monkeypatch.setenv("SPARMOE_VL_ROOT", str(checkout))
    monkeypatch.setenv("SPARMOE_VL_WORKSPACE", str(workspace))

    assert repository_root() == checkout.resolve()
    assert workspace_root() == workspace.resolve()


def test_workspace_defaults_to_repository_parent(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("SPARMOE_VL_ROOT", str(checkout))
    monkeypatch.delenv("SPARMOE_VL_WORKSPACE", raising=False)

    assert workspace_root() == tmp_path.resolve()
