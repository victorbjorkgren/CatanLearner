import React, {useEffect, useState} from "react";
import TaskButton, {TaskStates} from "../Utils/TaskButton";
import styles from "./Screens.module.css";
import {Screen, sendToServer, useGlobalState, useStateManager} from "../../misc/Misc";
import {ProgressBar} from "../Utils/ProgressBar";
import {CalibrationFrames, CalibrationSet} from "../PickCoordsTool/PickCoordTypes";
import {FramePickerTool} from "../FramePickerTool/FramePickerTool";
import {PickCoords} from "../PickCoordsTool/PickCoords";

interface CalibrationProps {
    gaitCycleProgress: number | undefined,
    gaitCycleImages: string[],
    calibrationFrames: CalibrationFrames,
    setCalibrationFrames: React.Dispatch<React.SetStateAction<CalibrationFrames>>,
    gaitcycleComplete: boolean
}

export const Calibration: React.FC<CalibrationProps> = (
    { gaitCycleProgress,
        gaitCycleImages,
        calibrationFrames,
        setCalibrationFrames,
        gaitcycleComplete}) => {

    const sm = useStateManager();
    let gs = useGlobalState();
    const hasPickedCoords = gs.pickCoordsStatus.state === TaskStates.Done;
    const hasPickedFrames = gs.pickCalibrationFramesStatus.state === TaskStates.Done;
    const [progressBarVisible, setProgressBarVisible] = useState<boolean>(false)

    const [pickFramesButton, setPickFramesButton] = useState<boolean>(false);
    const [pickCoordsButton, setPickCoordsButton] = useState<boolean>(false);
    const [pickFramesTool, setPickFramesTool] = useState<boolean>(false);
    const [pickCoordsTool, setPickCoordsTool] = useState<boolean>(false);


    const handlePickGaitFrames = (): void => {
        setPickFramesButton(false);
        setPickFramesTool(true);
    }
    const handlePickCoords = () => {
        setPickCoordsButton(false);
        setPickCoordsTool(true);
    }

    useEffect(() => {
        if (gaitcycleComplete)
        {
            setPickFramesButton(true);
            setProgressBarVisible(false);
        }
        else setProgressBarVisible(true);
    }, [gaitcycleComplete]);

    useEffect(() => {
        gs = sm.processes[sm.currentID]
    }, [sm.currentID]);



    const handleCalibration = (selectedFrames: number[]) => {
        sendToServer({
            key: "load_frame",
            value: selectedFrames[0],
            ID: sm.currentID
        });
        sendToServer({
            key: "stance_frame",
            value: selectedFrames[1],
            ID: sm.currentID
        });
        sendToServer({
            key: "flex_frame",
            value: selectedFrames[2],
            ID: sm.currentID
        });
        setCalibrationFrames({
            load: gaitCycleImages.find(image => image.endsWith(`color${selectedFrames[0]}.png`)) ?? "",
            stance: gaitCycleImages.find(image => image.endsWith(`color${selectedFrames[1]}.png`)) ?? "",
            flex: gaitCycleImages.find(image => image.endsWith(`color${selectedFrames[2]}.png`)) ?? "",
            static: calibrationFrames.static,
        });
        setPickFramesTool(false);
        setPickCoordsButton(true);
    }

    const handleSelection = (pointSets: CalibrationSet, isDone: boolean) => {
        if (isDone) {
            sendToServer({
                key: "calibration_points",
                value: pointSets,
                ID: sm.currentID
            })
            sm.setCurrentStep(Screen.ConfigScreen);
        }
        else{
            setPickCoordsTool(false);
            setPickFramesButton(true);
        }
    }


    return (
        <div className={styles.mainFrame}>
            Calibration
            <ProgressBar progress={gaitCycleProgress} visibility={progressBarVisible}/>
            {pickFramesButton &&<TaskButton
                taskFunc={handlePickGaitFrames}
                dispText="Pick Calibration Frames"
                taskStatus={{state: TaskStates.Ready, info:''}}
            />}
            {pickCoordsButton &&
            <TaskButton
                taskFunc={handlePickCoords}
                dispText="Pick Calibration Coordinates"
                taskStatus={{state: TaskStates.Ready, info:''}}
            />}
            {pickFramesTool &&
            <FramePickerTool
                images={gaitCycleImages}
                markerCount={3}
                handleSelection={handleCalibration}
                toolTips={["Pick LOAD frame", "Pick STANCE frame", "Pick FLEX frame"]}
            />
            }

            {pickCoordsTool &&
            <PickCoords
                image={calibrationFrames}
                handleSelection={handleSelection}
            />
            }
        </div>
    );
};