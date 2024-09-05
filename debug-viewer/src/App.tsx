// import logo from './logo.svg';
import './styles/App.css';
import React, {useEffect, useRef, useState} from 'react';
import SetupFolder from "./comps/Screens/SetupFolder";
import {Calibration} from "./comps/Screens/Calibration";
import {FramePickerTool} from "./comps/FramePickerTool/FramePickerTool";
import {Screen, sendToServer, useGlobalState, useStateManager} from "./misc/Misc";
import {TaskStates} from "./comps/Utils/TaskButton";
import {CalibrationFrames, CalibrationSet} from "./comps/PickCoordsTool/PickCoordTypes";
import {PickCoords} from "./comps/PickCoordsTool/PickCoords";
import {MainScreen} from "./comps/Screens/MainScreen";
import {DebugImageFolder} from "./comps/Screens/DebugImageFolder";
import ConfigScreen from "./comps/Screens/ConfigScreen";

interface GaitCycle {
    start: number,
    end: number
}

const App: React.FC = () => {
    const sm = useStateManager();
    const gs = useGlobalState();
    const [images, setImages] = useState<string[]>([]);
    const [gaitCycleImages, setGaitCycleImages] = useState<string[]>([]);
    const [staticGaitFrame, setStaticGaitFrame] = useState<string>();
    const [calibrationFrames, setCalibrationFrames] = useState<CalibrationFrames>({static: "", load: "", flex: "", stance:""});
    const [gaitCycle, setGaitCycle] = useState<GaitCycle | undefined>(undefined);
    const [calcStatus, setCalcStatus] = useState<string>("Not Initialized");
    const [calibrationStatus, setCalibrationStatus] = useState<string>("Not Initialized");
    const [gaitCycleProgress, setGaitCycleProgress] = useState<number | undefined>();
    const [kinematicsProgress, setKinematicsProgress] = useState<number | undefined>();
    const [gaitcycleComplete, setGaitcycleComplete] = useState<boolean>(false);


    // Utility to keep images updated inside functions
    const imageRef = useRef(images);
    useEffect(() => {
        imageRef.current = images;
    }, [images]);




    return (
        <DebugImageFolder
            calibrationStatus={calibrationStatus}
            calcStatus={calcStatus}
            calcProgress={kinematicsProgress}
        />
    )



}

export default App;
