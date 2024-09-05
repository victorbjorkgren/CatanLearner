import TaskButton, {TaskStates} from "../Utils/TaskButton";
import style from "./Screens.module.css"
import {Screen, useStateManager} from "../../misc/Misc";

export const MainScreen = () => {
    const sm = useStateManager();

    const handleRunFolderSetup = () => {
        sm.setCurrentStep(Screen.ConfigScreen)
    }

    return (
        <div className={style.mainBackground}>
            <div className={style.mainTitle}>
                MARKERLESS GAIT ANALYSIS
            </div>
            <div
                className={style.mainScreen}
            >
                <TaskButton
                    taskFunc={handleRunFolderSetup}
                    dispText={"Folder Selection"}
                    taskStatus={{state:TaskStates.Ready, info:''}}
                />
            </div>
        </div>
    );
};