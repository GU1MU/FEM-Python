"""只展示真实可用结果族的精简结果树。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

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

ResultRootKey = tuple[int, str]


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
    """按顶层作业或外部结果组织可用结果字段。"""

    fieldSelectionActivated = Signal(ScalarFieldSelection)
    fieldSelectionRouted = Signal(int, str, object, ScalarFieldSelection)
    runActivated = Signal(int, str)
    rootActionRequested = Signal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._activate_item)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._catalog: ResultCatalog | None = None
        self._section_point_labels: dict[int, str] = {}
        self._roots: dict[ResultRootKey, QTreeWidgetItem] = {}
        self._catalogs: dict[ResultRootKey, ResultCatalog] = {}
        self._root_kinds: dict[ResultRootKey, str] = {}
        self._root_signatures: dict[ResultRootKey, tuple[object, ...]] = {}
        self._keys_by_document: dict[int, set[ResultRootKey]] = {}
        self._active_run_ids: dict[int, str] = {}
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
        self._root_kinds.clear()
        self._root_signatures.clear()
        self._keys_by_document.clear()
        self._active_run_ids.clear()
        self._active_document_id = None
        self.clear()
        item = QTreeWidgetItem(["尚无分析结果"])
        self.addTopLevelItem(item)
        item.setData(0, ROLE_RESULT_KIND, "empty")

    @property
    def roots(self) -> dict[ResultRootKey, QTreeWidgetItem]:
        """Return job/result roots indexed by owner document and run identity."""

        return self._roots

    @property
    def catalogs(self) -> dict[ResultRootKey, ResultCatalog]:
        """Return the catalog currently projected for each job/result root."""

        return self._catalogs

    def set_active_document(self, document_id: int | None) -> None:
        self._active_document_id = None if document_id is None else int(document_id)
        if self._active_document_id is None:
            self._catalog = None
            return
        run_id = self._active_run_ids.get(self._active_document_id)
        key = None if run_id is None else (self._active_document_id, run_id)
        if key not in self._catalogs:
            key = next(
                (
                    candidate
                    for candidate in self._keys_by_document.get(
                        self._active_document_id,
                        (),
                    )
                    if candidate in self._catalogs
                ),
                None,
            )
        self._catalog = None if key is None else self._catalogs.get(key)

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
            document_id = self._active_document_id
            run_id = str(catalog.source.run_id)
            key = (document_id, run_id)
            current_root = self._roots.get(key)
            display_name = (
                current_root.text(0)
                if current_root is not None
                else step_name or "结果"
            )
            self._upsert_run_root(
                document_id,
                run_id,
                name=display_name,
                step_name=catalog.source.step_name,
                source_path=None,
                catalog=catalog,
                section_point_labels=section_point_labels,
                kind=self._root_kinds.get(key, "model"),
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
        self._root_kinds.clear()
        self._root_signatures.clear()
        self._keys_by_document.clear()
        self._active_run_ids.clear()
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
    ) -> tuple[QTreeWidgetItem, ...]:
        """Incrementally project each successful model run as a top-level root."""

        del display_name
        document_id = int(document_id)
        catalog = self._resolved_catalog(projection, catalog)
        labels = dict(section_point_labels or _projection_labels(projection))
        path = source_path or _projection_path(projection)
        runs = _successful_runs(projection)
        desired = {(document_id, str(getattr(run, "run_id", ""))) for run in runs}
        for key in tuple(self._keys_by_document.get(document_id, ())):
            if self._root_kinds.get(key) == "model" and key not in desired:
                self._remove_key(key, add_placeholder=False)

        source = None if catalog is None else catalog.source
        roots: list[QTreeWidgetItem] = []
        for run in runs:
            run_id = str(getattr(run, "run_id", ""))
            key = (document_id, run_id)
            run_catalog = (
                catalog
                if source is not None and str(source.run_id) == run_id
                else self._catalogs.get(key)
            )
            roots.append(
                self._upsert_run_root(
                    document_id,
                    run_id,
                    name=_run_label(run),
                    step_name=str(getattr(run, "step_name", "")),
                    source_path=path,
                    catalog=run_catalog,
                    section_point_labels=labels,
                    kind="model",
                )
            )
        self._ensure_empty_placeholder()
        return tuple(roots)

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
        """Incrementally project one external result as a top-level root."""

        document_id = int(document_id)
        catalog = self._resolved_catalog(projection, catalog)
        labels = dict(section_point_labels or _projection_labels(projection))
        path = source_path or _projection_path(projection)
        runs = _successful_runs(projection)
        if not runs:
            run_id = (
                str(catalog.source.run_id)
                if catalog is not None
                else str(getattr(projection, "displayed_result_run_id", "result"))
            )
            runs = (_archive_run(projection, run_id),)
        run = runs[0]
        run_id = str(getattr(run, "run_id", ""))
        for key in tuple(self._keys_by_document.get(document_id, ())):
            if key != (document_id, run_id):
                self._remove_key(key, add_placeholder=False)
        name = _display_name(
            display_name or (path if path is not None else None) or _run_label(run)
        )
        return self._upsert_run_root(
            document_id,
            run_id,
            name=name,
            step_name=str(getattr(run, "step_name", "")),
            source_path=path,
            catalog=catalog,
            section_point_labels=labels,
            kind="archive",
        )

    def remove_model_runs(self, document_id: int) -> bool:
        return self._remove_document(int(document_id))

    def remove_archive(self, document_id: int) -> bool:
        return self._remove_document(int(document_id))

    def set_active_source(self, source: object | None) -> None:
        """Select the first item carrying an exact result source identity."""

        if source is None:
            return
        stack = [self.topLevelItem(index) for index in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, ROLE_RESULT_SOURCE) == source:
                self.setCurrentItem(item)
                document_id = item.data(0, ROLE_DOCUMENT_ID)
                run_id = item.data(0, ROLE_RUN_ID)
                if document_id is not None and run_id:
                    normalized_document_id = int(document_id)
                    normalized_run_id = str(run_id)
                    self._active_run_ids[normalized_document_id] = normalized_run_id
                    if self._active_document_id in {None, normalized_document_id}:
                        self._catalog = self._catalogs.get(
                            (normalized_document_id, normalized_run_id)
                        )
                return
            stack.extend(item.child(index) for index in range(item.childCount()))

    def _show_context_menu(self, position: QPoint) -> None:
        """Expose lifecycle actions only for document result roots."""

        item = self.itemAt(position)
        if item is None or item.parent() is not None:
            return
        document_id = item.data(0, ROLE_DOCUMENT_ID)
        run_id = item.data(0, ROLE_RUN_ID)
        if document_id is None or not run_id:
            return
        document_id = int(document_id)
        run_id = str(run_id)
        key = (document_id, run_id)
        kind = self._root_kinds.get(key)
        if kind not in {"model", "archive"}:
            return
        self.setCurrentItem(item)
        menu = QMenu(self)
        activate = menu.addAction("激活")
        close = menu.addAction("关闭") if kind == "archive" else None
        chosen = menu.exec(self.viewport().mapToGlobal(position))
        if chosen is activate:
            self.runActivated.emit(document_id, run_id)
        elif close is not None and chosen is close:
            self.rootActionRequested.emit(document_id, "close")

    def _upsert_run_root(
        self,
        document_id: int,
        run_id: str,
        *,
        name: str,
        step_name: str,
        source_path: object | None,
        catalog: ResultCatalog | None,
        section_point_labels: Mapping[int, str] | None,
        kind: str,
    ) -> QTreeWidgetItem:
        if document_id < 0:
            raise ValueError("document_id must be non-negative")
        if not run_id:
            raise ValueError("run_id must not be empty")
        if type(catalog) is not ResultCatalog and catalog is not None:
            raise TypeError("catalog must be a ResultCatalog")
        labels = dict(section_point_labels or {})
        key = (document_id, run_id)
        old = self._roots.get(key)
        display_name = _display_name(name)
        signature = (
            display_name,
            step_name,
            None if source_path is None else str(source_path),
            id(catalog),
            tuple(labels.items()),
            kind,
        )
        if old is not None and self._root_signatures.get(key) == signature:
            if catalog is not None:
                self._active_run_ids[document_id] = run_id
                if self._active_document_id in {None, document_id}:
                    self._catalog = catalog
                self._section_point_labels = labels
            return old
        previous_index = self.indexOfTopLevelItem(old) if old is not None else -1
        if old is not None and previous_index >= 0:
            self.takeTopLevelItem(previous_index)

        root = QTreeWidgetItem([display_name])
        source = (
            None
            if catalog is None or str(catalog.source.run_id) != run_id
            else catalog.source
        )
        _set_identity(root, document_id, run_id, source, "run")
        if source_path is not None:
            root.setToolTip(0, str(source_path))
        default_item: QTreeWidgetItem | None = None
        if catalog is not None and source is not None:
            step, selected = self._catalog_step(
                source.step_name,
                catalog,
                document_id=document_id,
                run_id=run_id,
                source=source,
                section_point_labels=labels,
            )
            default_item = selected
        else:
            step = QTreeWidgetItem([step_name or "当前分析步"])
            _set_identity(step, document_id, run_id, None, "step")
        root.addChild(step)

        if old is None:
            self._remove_empty_placeholder()
        self._roots[key] = root
        self._root_kinds[key] = kind
        self._root_signatures[key] = signature
        self._keys_by_document.setdefault(document_id, set()).add(key)
        if catalog is None:
            self._catalogs.pop(key, None)
        else:
            self._catalogs[key] = catalog
            self._active_run_ids[document_id] = run_id
            if self._active_document_id in {None, document_id}:
                self._catalog = catalog
            self._section_point_labels = labels
        self.insertTopLevelItem(
            previous_index if previous_index >= 0 else self.topLevelItemCount(),
            root,
        )
        root.setExpanded(True)
        step.setExpanded(True)
        if default_item is not None and self._active_document_id in {None, document_id}:
            self.setCurrentItem(default_item)
        return root

    @staticmethod
    def _resolved_catalog(
        projection: object,
        catalog: ResultCatalog | None,
    ) -> ResultCatalog | None:
        if catalog is not None and type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be a ResultCatalog")
        resolved = _projection_catalog(projection) if catalog is None else catalog
        if resolved is not None and type(resolved) is not ResultCatalog:
            raise TypeError("projection catalog must be a ResultCatalog")
        return resolved

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

    def _remove_key(
        self,
        key: ResultRootKey,
        *,
        add_placeholder: bool = True,
    ) -> bool:
        document_id, run_id = key
        root = self._roots.pop(key, None)
        if root is None:
            return False
        index = self.indexOfTopLevelItem(root)
        if index >= 0:
            self.takeTopLevelItem(index)
        self._catalogs.pop(key, None)
        self._root_kinds.pop(key, None)
        self._root_signatures.pop(key, None)
        keys = self._keys_by_document.get(document_id)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._keys_by_document.pop(document_id, None)
        if self._active_run_ids.get(document_id) == run_id:
            replacement = next(iter(self._keys_by_document.get(document_id, ())), None)
            if replacement is None:
                self._active_run_ids.pop(document_id, None)
                if self._active_document_id == document_id:
                    self._catalog = None
            else:
                self._active_run_ids[document_id] = replacement[1]
                if self._active_document_id == document_id:
                    self._catalog = self._catalogs.get(replacement)
        if add_placeholder:
            self._ensure_empty_placeholder()
        return True

    def _remove_document(self, document_id: int) -> bool:
        keys = tuple(self._keys_by_document.get(document_id, ()))
        if not keys:
            return False
        for key in keys:
            self._remove_key(key, add_placeholder=False)
        if self._active_document_id == document_id:
            self._active_document_id = None
            self._catalog = None
        self._ensure_empty_placeholder()
        return True

    def _ensure_empty_placeholder(self) -> None:
        if self._roots:
            return
        for index in range(self.topLevelItemCount()):
            if self.topLevelItem(index).data(0, ROLE_RESULT_KIND) == "empty":
                return
        item = QTreeWidgetItem(["尚无分析结果"])
        item.setData(0, ROLE_RESULT_KIND, "empty")
        self.addTopLevelItem(item)

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
                self.topLevelItem(index) for index in range(self.topLevelItemCount())
            ]
        else:
            pending = [
                self._roots[key]
                for key in self._keys_by_document.get(int(document_id), ())
                if key in self._roots
            ]
        fallback: QTreeWidgetItem | None = None
        while pending:
            item = pending.pop(0)
            matches_document = document_id is None or item.data(
                0, ROLE_DOCUMENT_ID
            ) == int(document_id)
            matches_source = (
                source is None or item.data(0, ROLE_RESULT_SOURCE) == source
            )
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
            pending.extend(item.child(index) for index in range(item.childCount()))
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


def _successful_runs(projection: object) -> tuple[object, ...]:
    return tuple(
        run
        for run in getattr(projection, "runs", ())
        if bool(getattr(run, "has_result", False))
        or str(getattr(getattr(run, "status", None), "value", "")) == "succeeded"
    )


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
