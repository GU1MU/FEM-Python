from __future__ import annotations
from PySide6.QtWidgets import (
    QLabel,
    QMenu,
    QToolButton,
)
from fem.application import (
    ModelSession,
    describe_session_authoring,
)
from fem_gui.action_state import (
    ACTION_DESCRIPTORS,
    GuiActionContext,
    GuiActionKey,
    derive_action_availability,
)
from fem_gui.main_window import FEMMainWindow


def test_phase0_result_actions_replace_reload_close_in_project_surfaces(gui_application):
    descriptors = {item.key: item for item in ACTION_DESCRIPTORS}
    assert tuple(item.key for item in ACTION_DESCRIPTORS[:11]) == (
        GuiActionKey.OPEN,
        GuiActionKey.NEW_NATIVE,
        GuiActionKey.DELETE_MODEL,
        GuiActionKey.OPEN_PROJECT,
        GuiActionKey.SAVE_PROJECT,
        GuiActionKey.SAVE_PROJECT_AS,
        GuiActionKey.RELOAD,
        GuiActionKey.CLOSE,
        GuiActionKey.SAVE_RESULT,
        GuiActionKey.SAVE_RESULT_AS,
        GuiActionKey.OPEN_RESULT,
    )
    assert descriptors[GuiActionKey.SAVE_PROJECT_AS].text == "Save Model As..."
    assert descriptors[GuiActionKey.SAVE_PROJECT_AS].icon_name is None
    assert descriptors[GuiActionKey.SAVE_RESULT].text == "Save Results"
    assert descriptors[GuiActionKey.SAVE_RESULT].handler == "save_current_result"
    assert descriptors[GuiActionKey.SAVE_RESULT].icon_name == "save_result"
    assert descriptors[GuiActionKey.SAVE_RESULT_AS].text == "Save Results As..."
    assert descriptors[GuiActionKey.SAVE_RESULT_AS].icon_name is None
    assert descriptors[GuiActionKey.OPEN_RESULT].text == "Open Results"
    assert descriptors[GuiActionKey.OPEN_RESULT].handler == "open_result_file"
    assert descriptors[GuiActionKey.OPEN_RESULT].icon_name == "open_result"

    snapshot = ModelSession().snapshot()
    states = {
        item.key: item
        for item in derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(),
        )
    }
    assert not states[GuiActionKey.SAVE_RESULT].enabled
    assert "No successful results are available to save" in states[GuiActionKey.SAVE_RESULT].reason
    assert not states[GuiActionKey.SAVE_RESULT_AS].enabled
    assert (
        states[GuiActionKey.SAVE_RESULT_AS].reason
        == states[GuiActionKey.SAVE_RESULT].reason
    )
    assert states[GuiActionKey.OPEN_RESULT].enabled

    application = gui_application
    window = FEMMainWindow()
    file_menu = window.findChild(QMenu, "menuFile")
    assert file_menu is not None
    assert [
        action.objectName() for action in file_menu.actions()[:8]
    ] == [
        "action_new_native",
        "action_open_project",
        "action_save_project",
        "action_save_project_as",
        "action_open",
        "action_save_result",
        "action_save_result_as",
        "action_open_result",
    ]
    assert file_menu.actions()[8].isSeparator()
    assert file_menu.actions()[9] is window.actions["exit"]
    assert not window.actions["save_result"].icon().isNull()
    assert window.actions["save_project_as"].icon().isNull()
    assert window.actions["save_result_as"].icon().isNull()
    assert not window.actions["open_result"].icon().isNull()

    project_page = window.ribbon.stack.widget(
        [
            window.ribbon.tab_bar.tabText(index)
            for index in range(window.ribbon.tab_bar.count())
        ].index("Project")
    )
    file_label = next(
        label
        for label in project_page.findChildren(QLabel)
        if label.objectName() == "ribbonGroupTitle" and label.text() == "File"
    )
    file_group = file_label.parent()
    assert [
        button.defaultAction().objectName()
        for button in file_group.findChildren(QToolButton)
        if button.defaultAction() is not None
    ] == [
        "action_new_native",
        "action_delete_model",
        "action_open_project",
        "action_save_project",
        "action_open",
        "action_save_result",
        "action_open_result",
        "action_model_info",
    ]
    window.close()
    application.processEvents()
