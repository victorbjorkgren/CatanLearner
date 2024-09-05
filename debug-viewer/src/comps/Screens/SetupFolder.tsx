import React, {useEffect, useState} from "react";
import styles from "./Screens.module.css";
import TaskButton, {TaskStates} from "../Utils/TaskButton";
import {Screen, sendToServer, useGlobalState, useStateManager,} from "../../misc/Misc";
import {CalibrationFrames} from "../PickCoordsTool/PickCoordTypes";
import {FramePickerTool} from "../FramePickerTool/FramePickerTool";

interface SetupStepOneProps {
    images: string[];
    setImages: React.Dispatch<React.SetStateAction<string[]>>;
    calibrationImages: CalibrationFrames;
    setCalibrationImages: React.Dispatch<React.SetStateAction<CalibrationFrames>>;
}

export const SetupFolder: React.FC<SetupStepOneProps> = (
    {
        images,
        setImages,
        calibrationImages,
        setCalibrationImages
    }) => {
    const [isSidePopupVisible, setSidePopupVisible] = useState(false);

    const sm = useStateManager();
    const [current, setCurrent] = useState(sm.currentID);
    const [dynamic, setDynamic] = useState<boolean>(true);
    const [stat, setStat] = useState<boolean>(false);
    const [crop, setCrop] = useState<boolean>(false);
    const [side, setSide] = useState<boolean>(false);
    const [framePicker, setFramePicker] = useState<boolean>(false);


    useEffect(() => {
        setCurrent(sm.currentID);
    }, [sm.currentID]);


    const handleSelectDynamicFolder = async () => {
        const gs = sm.getCurrentGlobalState();

        if (window.electron && window.electron.selectFolder) {
            const [selectedImages, backgroundImage, folderPath] = await window.electron.selectFolder();
            if (selectedImages && selectedImages.length > 0) {
                setImages(selectedImages);
                console.log(`${selectedImages.length} images found`)

                sendToServer({
                    key: 'dynamic_acquisition_path',
                    value: folderPath,
                    ID: sm.currentID
                });

                setDynamic(false);
                setStat(true);
            }
        //     else if (!selectedImages) {
        //         gs.setSelectDynFolderStatus({
        //             state: TaskStates.Ready,
        //             info: "No folder selected"
        //         });
        //     }
        //     else if (selectedImages && selectedImages.length === 0){
        //         gs.setSelectDynFolderStatus({
        //             state:TaskStates.Error,
        //             info: "No .png images in the selected folder"
        //         });
        //     }
        //     else{
        //         gs.setSelectDynFolderStatus({
        //             state:TaskStates.Error,
        //             info: "Unknown error"
        //         });
        //     }
        // } else {
        //     gs.setSelectDynFolderStatus({
        //         state:TaskStates.Error,
        //         info: "SOURCE CODE ERROR: FUNCTION window.electron.selectFolder NOT FOUND"
        //     });
        //     console.error('window.electron.selectFolder is not defined');
        }

    };

    const handleSelectStaticFrame = async () => {
        const gs = sm.getCurrentGlobalState();
        if (window.electron && window.electron.selectFolder) {
            const [staticImages, _, staticFolder] = await window.electron.selectFolder();
            if (staticImages && staticImages.length > 0) {
                sendToServer({
                    key: 'static_acquisition_path',
                    value: staticFolder,
                    ID: sm.currentID
                });
                setCalibrationImages({
                    static: staticImages[0],
                    stance: calibrationImages.stance,
                    flex: calibrationImages.flex,
                    load: calibrationImages.load,
                });
                setStat(false);
                setCrop(true);
            }
        //     else if (!staticImages)
        //         gs.setSelectStaticFrameStatus({
        //             state: TaskStates.Ready,
        //             info: "No folder chosen"
        //         })
        //     else if (staticImages && staticImages.length == 0)
        //         gs.setSelectStaticFrameStatus({
        //             state: TaskStates.Error,
        //             info: "No .png images in the selected folder"
        //         })
        //     else
        //         gs.setSelectStaticFrameStatus({
        //             state:TaskStates.Error,
        //             info: "Unknown error"
        //         });
        // } else {
        //     gs.setSelectStaticFrameStatus({
        //         state:TaskStates.Error,
        //         info: "SOURCE CODE ERROR: FUNCTION window.electron.selectFile NOT FOUND"
        //     });
        //     console.error('window.electron.selectFile is not defined');
        }
    }

    const handleSelectSide = () => {
        setSidePopupVisible(true);
    }

    const handleLeft = () => {
        setSidePopupVisible(false);
        sendToServer({key: 'side', value:'L', ID: sm.currentID});
        sm.setCurrentStep(Screen.CalibrationMain);
    }

    const handleRight = () => {
        setSidePopupVisible(false);
        sendToServer({key: 'side', value:'R', ID: sm.currentID});
        sm.setCurrentStep(Screen.CalibrationMain);
    }

    const cropDynamicFrames = () => {
        setCrop(false);
        setFramePicker(true);
    }

    const handleCrop = (selectedFrames: number[]) => {
        const min = Math.min(...selectedFrames)
        const max = Math.max(...selectedFrames)
        sendToServer({
            key: "first_frame",
            value: min,
            ID: sm.currentID
        });
        sendToServer({
            key: "last_frame",
            value: max,
            ID: sm.currentID
        });
        sm.setCurrentStep(Screen.SetupFolder);
        setFramePicker(false);
        setSide(true);
    }

    return (
        <div>
            <div className={`${styles.mainFrame} ${isSidePopupVisible ? 'blur' : ''}`}>
                Step - File Choice Settings
                {dynamic && <TaskButton
                    taskFunc={handleSelectDynamicFolder}
                    dispText="Select Dynamic Acquisition Folder"
                    taskStatus={{state: TaskStates.Ready, info: ""}}
                />}
                {stat &&
                <TaskButton
                    taskFunc={handleSelectStaticFrame}
                    dispText="Select Static Frame"
                    taskStatus={{state: TaskStates.Ready, info: ""}}
                />}
                {crop &&
                <TaskButton
                    taskFunc={cropDynamicFrames}
                    dispText="Crop Dynamic Frames"
                    taskStatus={{state: TaskStates.Ready, info: ""}}
                />}
                {side &&
                <TaskButton
                    taskFunc={handleSelectSide}
                    dispText="Select Side"
                    taskStatus={{state: TaskStates.Ready, info: ""}}
                />}
                {framePicker &&
                <FramePickerTool
                    images={images}
                    markerCount={2}
                    handleSelection={handleCrop}
                    toolTips={["Choose Range to Analyze"]}
                />}

            </div>
            {isSidePopupVisible && (
                <div className={styles.popup}>
                    <TaskButton taskFunc={handleLeft} dispText={"Left"} taskStatus={{state: TaskStates.Ready, info: ""}} />
                    <TaskButton taskFunc={handleRight} dispText={"Right"} taskStatus={{state: TaskStates.Ready, info: ""}} />
                </div>
            )}
        </div>
    );
};


export default SetupFolder;
