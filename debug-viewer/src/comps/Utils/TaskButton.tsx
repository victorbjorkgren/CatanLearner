import React, {useState} from 'react';
import styles from "./Buttons.module.css";

export enum TaskStates {
    NotReady,
    Ready,
    Active,
    Done,
    Error,
}

export interface TaskStatus {
    state: TaskStates;
    info: string;
}

interface TaskButtonProps {
    taskFunc: () => void;
    dispText: string;
    taskStatus: TaskStatus;
}

export const TaskButton: React.FC<TaskButtonProps> = ({ taskFunc, dispText, taskStatus }) => {
    let buttonStyle;

    switch (taskStatus.state) {
        case TaskStates.NotReady:
            buttonStyle = styles.notReadyButton;
            break;
        case TaskStates.Ready:
            buttonStyle = styles.readyButton;
            break;
        case TaskStates.Active:
            buttonStyle = styles.activeButton;
            break;
        case TaskStates.Done:
            buttonStyle = styles.doneButton;
            break;
        case TaskStates.Error:
            buttonStyle = styles.errorButton;
            break;
        default:
            buttonStyle = styles.button; // fallback to default style
    }

    const handleClick = () => {
        if (taskStatus.state !== TaskStates.NotReady) {
            taskFunc();
        }
    }

    return (
        <div>
            <button
                onClick={handleClick}
                className={`${styles.button} ${buttonStyle}`}
                disabled={taskStatus.state === TaskStates.NotReady}
            >
                {dispText}
            </button>
            {taskStatus.info && <div className={styles.infoText}>{taskStatus.info}</div>}
        </div>
    )
}

export default TaskButton;