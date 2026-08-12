"""只展示真实可用结果族的精简结果树。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from fem.application.results import (
    FieldAvailability,
    FieldState,
    ResultCatalog,
    ScalarFieldSelection,
)
from fem_gui.result_presentation import (
    result_field_is_beam_section,
    result_field_position_label,
    visible_result_fields,
)

ROLE_SELECTION = int(Qt.ItemDataRole.UserRole)
ROLE_MATERIALIZATION_KEY = ROLE_SELECTION + 1
ROLE_FIELD_STATE = ROLE_SELECTION + 2
ROLE_DOCUMENT_ID = ROLE_FIELD_STATE + 1
ROLE_RUN_ID = ROLE_DOCUMENT_ID + 1
ROLE_RESULT_SOURCE = ROLE_RUN_ID + 1
ROLE_RESULT_KIND = ROLE_RESULT_SOURCE + 1


_FIELD_LABELS = {
    "result.field.u.node": "位移 U",
    "result.field.ur.node": "转角 UR",
    "result.field.rf.node": "反力 RF",
    "result.field.rm.node": "反力矩 RM",
    "result.field.sf.integration_point": "截面力 SF（积分点）",
    "result.field.sm.integration_point": "截面矩 SM（积分点）",
    "result.field.le.centroid": "对数应变 LE",
    "result.field.s.element_nodal": "应力 S",
}


class ResultTree(QTreeWidget):
    """按位移、反力和应力组织当前单步结果。"""

    fieldSelectionActivated = Signal(ScalarFieldSelection)
    fieldSelectionRouted = Signal(int, str, object, ScalarFieldSelection)
    runActivated = Signal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._activate_item)
        self._catalog: ResultCatalog | None = None
        self._section_point_labels: dict[int, str] = {}
        self._roots: dict[int, QTreeWidgetItem] = {}
        self._catalogs: dict[int, ResultCatalog] = {}
        self._root_sources: dict[int, object] = {}
        self._root_kinds: dict[int, str] = {}
        self._active_document_id: int | None = None
        self.clear_result()

    @property
    def catalog(self) -> ResultCatalog | None:
        """Return the exact immutable catalog installed in this view."""

        return self._catalog

    def clear_result(self) -> None:
        self._catalog = None
        self._section_point_labels = {}
        self._roots.clear()
        self._catalogs.clear()
        self._root_sources.clear()
        self._root_kinds.clear()
        self._active_document_id = None
        self.clear()
        item = QTreeWidgetItem(["尚无分析结果"])
        self.addTopLevelItem(item)
        item.setData(0, ROLE_RESULT_KIND, "empty")

    @property
    def roots(self) -> dict[int, QTreeWidgetItem]:
        """Return document roots indexed by integer workspace identity."""

        return self._roots

    @property
    def catalogs(self) -> dict[int, ResultCatalog]:
        """Return the catalog currently projected for each document."""

        return self._catalogs

    def set_active_document(self, document_id: int | None) -> None:
        self._active_document_id = (
            None if document_id is None else int(document_id)
        )
        self._catalog = (
            None
            if self._active_document_id is None
            else self._catalogs.get(self._active_document_id)
        )

    def set_catalog(
        self,
        step_name: str,
        catalog: ResultCatalog,
        *,
        section_point_labels: Mapping[int, str] | None = None,
    ) -> None:
        """Populate the tree from one immutable application catalog."""

        if type(step_name) is not str:
            raise TypeError("step_name must be a string")
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be a ResultCatalog")

        # A caller from the pre-Phase-4 job manager may still use
        # ``set_catalog``.  Once a workspace document is active, route that
        # update through the indexed root instead of clearing all documents.
        if self._active_document_id is not None:
            from types import SimpleNamespace

            document_id = self._active_document_id
            current_root = self._roots.get(document_id)
            display_name = (
                current_root.text(0)
                if current_root is not None
                else "结果"
            )
            projection = SimpleNamespace(
                model_name=display_name,
                runs=(
                    SimpleNamespace(
                        run_id=catalog.source.run_id,
                        name=step_name,
                        step_name=catalog.source.step_name,
                        has_result=True,
                    ),
                ),
            )
            self._upsert_document_root(
                document_id,
                projection,
                display_name=display_name,
                source_path=None,
                catalog=catalog,
                section_point_labels=section_point_labels,
                kind=self._root_kinds.get(document_id, "model"),
            )
            self._catalog = catalog
            return

        # Legacy single-document compatibility. Multi-result callers use the
        # incremental APIs below and never call ``clear``.
        preserve_expansion = (
            self._catalog is not None and self._catalog.source == catalog.source
        )
        expanded_paths = self._expanded_item_paths() if preserve_expansion else None
        self.clear()
        self._roots.clear()
        self._catalogs.clear()
        self._root_sources.clear()
        self._root_kinds.clear()
        self._catalog = None
        self._section_point_labels = dict(section_point_labels or {})
        root = QTreeWidgetItem(["分析结果"])
        step = QTreeWidgetItem([step_name or "当前分析步"])
        root.addChild(step)

        default_item: QTreeWidgetItem | None = None
        beam_stress_item: QTreeWidgetItem | None = None
        for availability in visible_result_fields(catalog.fields):
            if result_field_is_beam_section(availability.descriptor.field_id):
                if beam_stress_item is None:
                    beam_stress_item = QTreeWidgetItem(["应力 S"])
                    step.addChild(beam_stress_item)
                field_item, selected_component = self._catalog_field_item(
                    availability,
                    catalog.default_selection,
                    field_label=result_field_position_label(
                        availability.descriptor.field_id,
                        section_point_labels=self._section_point_labels,
                    ),
                )
                beam_stress_item.addChild(field_item)
                if selected_component is not None:
                    default_item = selected_component
                continue
            field_item, selected_component = self._catalog_field_item(
                availability,
                catalog.default_selection,
            )
            step.addChild(field_item)
            if selected_component is not None:
                default_item = selected_component

        self.addTopLevelItem(root)
        if expanded_paths is None:
            root.setExpanded(True)
            step.setExpanded(True)
            parent = None if default_item is None else default_item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
        else:
            self._restore_expanded_item_paths(expanded_paths)
        if default_item is not None:
            self.setCurrentItem(default_item)
        self._catalog = catalog

    def upsert_model_runs(
        self,
        document_id: int,
        projection: object,
        *,
        display_name: str | None = None,
        source_path: object | None = None,
        catalog: ResultCatalog | None = None,
        section_point_labels: Mapping[int, str] | None = None,
    ) -> QTreeWidgetItem:
        """Incrementally replace one model's successful-run result root."""

        return self._upsert_document_root(
            int(document_id),
            projection,
            display_name=display_name,
            source_path=source_path,
            catalog=catalog,
            section_point_labels=section_point_labels,
            kind="model",
        )

    def upsert_archive(
        self,
        document_id: int,
        projection: object,
        *,
        display_name: str | None = None,
        source_path: object | None = None,
        catalog: ResultCatalog | None = None,
        section_point_labels: Mapping[int, str] | None = None,
    ) -> QTreeWidgetItem:
        """Incrementally replace one external result-only root."""

        return self._upsert_document_root(
            int(document_id),
            projection,
            display_name=display_name,
            source_path=source_path,
            catalog=catalog,
            section_point_labels=section_point_labels,
            kind="archive",
        )

    def remove_model_runs(self, document_id: int) -> bool:
        return self._remove_root(int(document_id))

    def remove_archive(self, document_id: int) -> bool:
        return self._remove_root(int(document_id))

    def set_active_source(self, source: object | None) -> None:
        """Select the first item carrying an exact result source identity."""

        if source is None:
            return
        stack = [
            self.topLevelItem(index)
            for index in range(self.topLevelItemCount())
        ]
        while stack:
            item = stack.pop()
            if item.data(0, ROLE_RESULT_SOURCE) == source:
                self.setCurrentItem(item)
                return
            stack.extend(item.child(index) for index in range(item.childCount()))

    def _upsert_document_root(
        self,
        document_id: int,
        projection: object,
        *,
        display_name: str | None,
        source_path: object | None,
        catalog: ResultCatalog | None,
        section_point_labels: Mapping[int, str] | None,
        kind: str,
    ) -> QTreeWidgetItem:
        if document_id < 0:
            raise ValueError("document_id must be non-negative")
        if catalog is not None and type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be a ResultCatalog")
        if catalog is None:
            catalog = _projection_catalog(projection)
        if catalog is not None and type(catalog) is not ResultCatalog:
            raise TypeError("projection catalog must be a ResultCatalog")
        labels = dict(section_point_labels or _projection_labels(projection))
        path = source_path or _projection_path(projection)
        name = _display_name(display_name or _projection_name(projection) or "Result")
        if display_name is None and path is not None:
            name = _display_name(path)

        runs = tuple(
            run
            for run in getattr(projection, "runs", ())
            if bool(getattr(run, "has_result", False))
            or str(getattr(getattr(run, "status", None), "value", ""))
            == "succeeded"
        )
        if kind == "archive" and not runs:
            run_id = (
                str(catalog.source.run_id)
                if catalog is not None
                else str(getattr(projection, "displayed_result_run_id", "result"))
            )
            runs = (_archive_run(projection, run_id),)

        old = self._roots.get(document_id)
        previous_index = self.indexOfTopLevelItem(old) if old is not None else -1
        if old is not None and previous_index >= 0:
            self.takeTopLevelItem(previous_index)

        root = QTreeWidgetItem([name])
        source = None if catalog is None else catalog.source
        _set_identity(root, document_id, None, source, kind)
        if path is not None:
            root.setToolTip(0, str(path))
        default_item: QTreeWidgetItem | None = None
        for run in runs:
            run_id = str(getattr(run, "run_id", ""))
            run_item = QTreeWidgetItem([_run_label(run)])
            run_source = (
                source if source is not None and str(source.run_id) == run_id else None
            )
            _set_identity(run_item, document_id, run_id, run_source, "run")
            root.addChild(run_item)
            if catalog is None or run_source is None:
                continue
            step, selected = self._catalog_step(
                source.step_name,
                catalog,
                document_id=document_id,
                run_id=run_id,
                source=source,
                section_point_labels=labels,
            )
            run_item.addChild(step)
            if selected is not None:
                default_item = selected

        if old is None:
            self._remove_empty_placeholder()
        self._roots[document_id] = root
        self._root_kinds[document_id] = kind
        if catalog is None:
            self._catalogs.pop(document_id, None)
            self._root_sources.pop(document_id, None)
            if self._active_document_id in {None, document_id}:
                self._catalog = None
        else:
            self._catalogs[document_id] = catalog
            self._root_sources[document_id] = source
            if self._active_document_id in {None, document_id}:
                self._catalog = catalog
            self._section_point_labels = labels
        self.insertTopLevelItem(
            previous_index if previous_index >= 0 else self.topLevelItemCount(),
            root,
        )
        root.setExpanded(True)
        if (
            default_item is not None
            and self._active_document_id in {None, document_id}
        ):
            self.setCurrentItem(default_item)
        return root

    def _catalog_step(
        self,
        step_name: str,
        catalog: ResultCatalog,
        *,
        document_id: int,
        run_id: str,
        source: object,
        section_point_labels: Mapping[int, str],
    ) -> tuple[QTreeWidgetItem, QTreeWidgetItem | None]:
        step = QTreeWidgetItem([step_name or "当前分析步"])
        _set_identity(step, document_id, run_id, source, "step")
        default_item: QTreeWidgetItem | None = None
        beam_stress_item: QTreeWidgetItem | None = None
        for availability in visible_result_fields(catalog.fields):
            if result_field_is_beam_section(availability.descriptor.field_id):
                if beam_stress_item is None:
                    beam_stress_item = QTreeWidgetItem(["S"])
                    _set_identity(
                        beam_stress_item,
                        document_id,
                        run_id,
                        source,
                        "group",
                    )
                    step.addChild(beam_stress_item)
                field_item, selected_component = self._catalog_field_item(
                    availability,
                    catalog.default_selection,
                    field_label=result_field_position_label(
                        availability.descriptor.field_id,
                        section_point_labels=section_point_labels,
                    ),
                )
                _annotate_identity(field_item, document_id, run_id, source)
                beam_stress_item.addChild(field_item)
                if selected_component is not None:
                    default_item = selected_component
                continue
            field_item, selected_component = self._catalog_field_item(
                availability,
                catalog.default_selection,
            )
            _annotate_identity(field_item, document_id, run_id, source)
            step.addChild(field_item)
            if selected_component is not None:
                default_item = selected_component
        return step, default_item

    def _remove_root(self, document_id: int) -> bool:
        was_active = self._active_document_id == document_id
        root = self._roots.pop(document_id, None)
        if root is None:
            return False
        index = self.indexOfTopLevelItem(root)
        if index >= 0:
            self.takeTopLevelItem(index)
        self._catalogs.pop(document_id, None)
        self._root_sources.pop(document_id, None)
        self._root_kinds.pop(document_id, None)
        if was_active:
            self._active_document_id = None
            self._catalog = None
        if not self._roots:
            item = QTreeWidgetItem(["暂无分析结果"])
            item.setData(0, ROLE_RESULT_KIND, "empty")
            self.addTopLevelItem(item)
        return True

    def _remove_empty_placeholder(self) -> None:
        for index in range(self.topLevelItemCount() - 1, -1, -1):
            item = self.topLevelItem(index)
            if item.data(0, ROLE_RESULT_KIND) == "empty":
                self.takeTopLevelItem(index)

    @staticmethod
    def _catalog_field_item(
        availability: FieldAvailability,
        default_selection: ScalarFieldSelection | None,
        *,
        field_label: str | None = None,
    ) -> tuple[QTreeWidgetItem, QTreeWidgetItem | None]:
        descriptor = availability.descriptor
        if field_label is None:
            field_label = _FIELD_LABELS.get(
                descriptor.label_key,
                descriptor.label_key,
            )
        field_item = QTreeWidgetItem([field_label])
        field_selection = ScalarFieldSelection(
            availability.key,
            descriptor.default_component,
        )
        _set_typed_item_data(
            field_item,
            availability,
            field_selection,
        )

        selected_component: QTreeWidgetItem | None = None
        for component in descriptor.columns:
            selection = ScalarFieldSelection(availability.key, component)
            component_item = QTreeWidgetItem([component])
            _set_typed_item_data(
                component_item,
                availability,
                selection,
            )
            field_item.addChild(component_item)
            if selection == default_selection:
                selected_component = component_item

        if availability.state is FieldState.UNAVAILABLE:
            _disable_item(field_item)
            for index in range(field_item.childCount()):
                _disable_item(field_item.child(index))
        return field_item, selected_component

    def select_selection(
        self,
        selection: ScalarFieldSelection,
        *,
        document_id: int | None = None,
        source: object | None = None,
    ) -> bool:
        """Select the exact catalog component without rebuilding the tree."""

        item = self._selection_item(
            selection,
            document_id=document_id,
            source=source,
        )
        if item is None:
            return False
        self.setCurrentItem(item)
        return True

    def _expanded_item_paths(self) -> set[tuple[str, ...]]:
        expanded: set[tuple[str, ...]] = set()

        def visit(item: QTreeWidgetItem, parent_path: tuple[str, ...]) -> None:
            path = (*parent_path, item.text(0))
            if item.isExpanded():
                expanded.add(path)
            for index in range(item.childCount()):
                visit(item.child(index), path)

        for index in range(self.topLevelItemCount()):
            visit(self.topLevelItem(index), ())
        return expanded

    def _restore_expanded_item_paths(
        self,
        expanded_paths: set[tuple[str, ...]],
    ) -> None:
        def visit(item: QTreeWidgetItem, parent_path: tuple[str, ...]) -> None:
            path = (*parent_path, item.text(0))
            item.setExpanded(path in expanded_paths)
            for index in range(item.childCount()):
                visit(item.child(index), path)

        for index in range(self.topLevelItemCount()):
            visit(self.topLevelItem(index), ())

    def has_selection(
        self,
        selection: ScalarFieldSelection,
        *,
        document_id: int | None = None,
        source: object | None = None,
    ) -> bool:
        """Return whether the exact component is present without changing UI."""

        return (
            self._selection_item(
                selection,
                document_id=document_id,
                source=source,
            )
            is not None
        )

    def _selection_item(
        self,
        selection: ScalarFieldSelection,
        *,
        document_id: int | None = None,
        source: object | None = None,
    ) -> QTreeWidgetItem | None:
        if type(selection) is not ScalarFieldSelection:
            raise TypeError("selection must be a ScalarFieldSelection")
        if document_id is None:
            pending = [
                self.topLevelItem(index)
                for index in range(self.topLevelItemCount())
            ]
        else:
            root = self._roots.get(int(document_id))
            pending = [] if root is None else [root]
        fallback: QTreeWidgetItem | None = None
        while pending:
            item = pending.pop(0)
            matches_document = (
                document_id is None
                or item.data(0, ROLE_DOCUMENT_ID) == int(document_id)
            )
            matches_source = source is None or item.data(0, ROLE_RESULT_SOURCE) == source
            if (
                matches_document
                and matches_source
                and item.data(0, ROLE_SELECTION) == selection
            ):
                if item.childCount() == 0:
                    fallback = item
                    break
                if fallback is None:
                    fallback = item
            pending.extend(
                item.child(index)
                for index in range(item.childCount())
            )
        return fallback

    def _activate_item(self, item: QTreeWidgetItem) -> None:
        if item.data(0, ROLE_RESULT_KIND) == "run":
            document_id = item.data(0, ROLE_DOCUMENT_ID)
            run_id = item.data(0, ROLE_RUN_ID)
            if document_id is not None and run_id:
                self.runActivated.emit(int(document_id), str(run_id))
            return
        selection = item.data(0, ROLE_SELECTION)
        state = item.data(0, ROLE_FIELD_STATE)
        if (
            type(selection) is ScalarFieldSelection
            and state != FieldState.UNAVAILABLE.value
        ):
            self.fieldSelectionActivated.emit(selection)
            document_id = item.data(0, ROLE_DOCUMENT_ID)
            run_id = item.data(0, ROLE_RUN_ID)
            source = item.data(0, ROLE_RESULT_SOURCE)
            if document_id is not None:
                self.fieldSelectionRouted.emit(
                    int(document_id),
                    str(run_id or ""),
                    source,
                    selection,
                )


def _set_typed_item_data(
    item: QTreeWidgetItem,
    availability: FieldAvailability,
    selection: ScalarFieldSelection,
) -> None:
    item.setData(0, ROLE_SELECTION, selection)
    item.setData(0, ROLE_MATERIALIZATION_KEY, availability.key)
    item.setData(0, ROLE_FIELD_STATE, availability.state.value)


def _set_identity(
    item: QTreeWidgetItem,
    document_id: int,
    run_id: str | None,
    source: object | None,
    kind: str,
) -> None:
    item.setData(0, ROLE_DOCUMENT_ID, int(document_id))
    item.setData(0, ROLE_RUN_ID, run_id)
    item.setData(0, ROLE_RESULT_SOURCE, source)
    item.setData(0, ROLE_RESULT_KIND, kind)


def _annotate_identity(
    item: QTreeWidgetItem,
    document_id: int,
    run_id: str,
    source: object | None,
) -> None:
    _set_identity(item, document_id, run_id, source, "field")
    for index in range(item.childCount()):
        _annotate_identity(item.child(index), document_id, run_id, source)


def _projection_catalog(projection: object) -> ResultCatalog | None:
    for candidate in (
        getattr(projection, "catalog", None),
        getattr(projection, "result_catalog", None),
    ):
        if type(candidate) is ResultCatalog:
            return candidate
    provider = getattr(projection, "result_provider", None)
    if provider is not None:
        candidate = provider.catalog()
        if type(candidate) is ResultCatalog:
            return candidate
    return None


def _projection_labels(projection: object) -> Mapping[int, str]:
    labels = getattr(projection, "section_point_labels", None)
    return labels if isinstance(labels, Mapping) else {}


def _projection_name(projection: object) -> str | None:
    for name in (
        getattr(projection, "display_name", None),
        getattr(projection, "model_name", None),
        getattr(getattr(projection, "origin", None), "model_name", None),
    ):
        if name:
            return str(name)
    return None


def _projection_path(projection: object) -> object | None:
    return getattr(projection, "source_path", None) or getattr(projection, "path", None)


def _display_name(value: object) -> str:
    from pathlib import Path

    text = str(value)
    suffix = Path(text).suffix.casefold()
    if suffix in {".femres", ".fempy", ".femproj", ".inp", ".json"}:
        return Path(text).stem
    return text


def _run_label(run: object) -> str:
    name = str(getattr(run, "name", "") or getattr(run, "run_id", "run"))
    return name


def _archive_run(projection: object, run_id: str) -> object:
    from types import SimpleNamespace

    origin = getattr(projection, "origin", None)
    return SimpleNamespace(
        run_id=run_id,
        name=getattr(origin, "run_name", None) or run_id,
        step_name=getattr(origin, "step_name", None)
        or getattr(getattr(projection, "source", None), "step_name", ""),
        has_result=True,
    )


def _disable_item(item: QTreeWidgetItem) -> None:
    item.setFlags(
        item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable
    )
