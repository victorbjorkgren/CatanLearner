import TaskButton, {TaskStates} from "../Utils/TaskButton";
import {Screen, useGlobalState, useStateManager} from "../../misc/Misc";
import {ProgressBar} from "../Utils/ProgressBar";
import {useEffect, useState} from "react";
import {FramePickerTool} from "../FramePickerTool/FramePickerTool";
import path from "path";

interface ReportStepProps {
    calibrationStatus: string;
    calcStatus: string;
    calcProgress: number | undefined
}

async function getDebugFiles() {
    if (window.electron && window.electron.getDebugFiles)
        return await window.electron.getDebugFiles();
}

export const DebugImageFolder: React.FC<ReportStepProps> = ({calibrationStatus, calcStatus, calcProgress} ) => {
    const gs = useGlobalState();
    const sm = useStateManager();
    const [debugFiles, setDebugFiles] = useState(['']);
    const [debugDirectories, setDebugDirectories] = useState(['']);
    const [selectedFiles, setSelectedFiles] = useState(['']);
    const [isVisible, setIsVisible] = useState(false);

    const handleBack = () => {
        sm.setCurrentStep(Screen.Main)
    }

    const handleDebug = () => {
        getDebugFiles().then((files) =>{
            if (files){
                setDebugFiles(files[0]);
                setDebugDirectories(files[1]);
            }
        });
    }

    const handleSelect = (e: { target: { value: any; }; }) => {
        let selectedDirectory = e.target.value;
        let selection = debugFiles.filter((element) => {
            return element.includes(selectedDirectory);
        });

        if (selection) {
            selection.sort((a, b) => {
                // Function to extract the file name from the full path
                const extractFileName = (fullPath: string) => {
                    const parts = fullPath.split(/[\\/]/); // Split by both / and \
                    return parts[parts.length - 1]; // Take the last part as the file name
                };

                // Function to extract all numbers from the string using regex
                const extractNumbers = (str: string) => {
                    const fileName = extractFileName(str); // Extract just the file name
                    // Regular expression to match all sequences of digits in the file name
                    const numberMatches = fileName.match(/\d+/g);
                    // Convert matched strings to numbers and return them (default to 0 if no matches)
                    return numberMatches ? numberMatches.map(Number) : [0];
                };

                const numsA = extractNumbers(a);
                const numsB = extractNumbers(b);

                // Log extracted numbers for debugging
                console.log(`Comparing ${a} -> ${numsA} with ${b} -> ${numsB}`);

                // Compare each number group sequentially
                for (let i = 0; i < Math.max(numsA.length, numsB.length); i++) {
                    const numA = numsA[i] || 0; // Default to 0 if one array is shorter
                    const numB = numsB[i] || 0;
                    if (numA !== numB) {
                        return numA - numB; // Return the comparison result
                    }
                }
                return 0; // If all compared numbers are equal, return 0 (no sorting needed)
            });

            console.log("Sorted Selection:", selection);
            setSelectedFiles(selection);
        }
    };

    return (
        <div>
            <div className={'horizontal-buttons'}>
                <TaskButton
                    taskFunc={handleDebug}
                    dispText={"Set Debug Folder"}
                    taskStatus={{state: TaskStates.Ready, info: ""}}
                />
            </div>
            <select className="directory-select" onChange={handleSelect}>
                {debugDirectories.map((directory =>
                <option value={directory}>{directory}</option>))}
            </select>
            <FramePickerTool
                images={selectedFiles}
                markerCount={1}
                handleSelection={()=>{}}
                toolTips={['']}
            />
        </div>
    );
};