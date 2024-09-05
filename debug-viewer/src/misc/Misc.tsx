import React, { createContext, ReactNode, useContext, useState } from "react";
import { TaskStates, TaskStatus } from "../comps/Utils/TaskButton";

export interface ServerResponse {
    status: string;
    message: string;
    ID: number;
}

export interface DataToSend {
    key: string;
    value: any;
    ID: number;
}

export enum Screen {
    Main,
    SetupFolder,
    CropImageRange,
    CalibrationMain,
    CalibrationPickFrames,
    CalibrationPickCoords,
    Results,
    ConfigScreen
}



interface GlobalState {
    folderSelectionStatus: TaskStatus,
    setFolderSelectionStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    selectDynFolderStatus: TaskStatus,
    setSelectDynFolderStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    selectStaticFrameStatus: TaskStatus,
    setSelectStaticFrameStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    selectSideStatus: TaskStatus,
    setSelectSideStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    cropDynamicFramesStatus: TaskStatus,
    setCropDynamicFramesStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    gaitCycleStatus: TaskStatus,
    setGaitCycleStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    pickCalibrationFramesStatus: TaskStatus,
    setPickCalibrationFramesStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    pickCoordsStatus: TaskStatus,
    setPickCoordsStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,
    runKinematicsStatus: TaskStatus,
    setRunKinematicsStatus: React.Dispatch<React.SetStateAction<TaskStatus>>,

    ID: number,
    displayName: string,
    setDisplayName: React.Dispatch<React.SetStateAction<string>>
}

interface StateManager {
    processes: GlobalState[],
    currentID: number,
    setCurrentID: React.Dispatch<React.SetStateAction<number>>,
    currentStep: Screen,
    setCurrentStep: React.Dispatch<React.SetStateAction<Screen>>,
    createAndAddProcess: () => GlobalState,
    removeProcess: (ID: number) => void,
    setProcesses: React.Dispatch<React.SetStateAction<GlobalState[]>>,
    updateProcesses: (newState: GlobalState) => void,
    getCurrentGlobalState: () => GlobalState,
}

const GlobalStateContext = createContext<StateManager | undefined>(undefined);

export const GlobalStateProvider: React.FC<{ children: ReactNode }> = ({ children }) => {

    const [processes, setProcesses] = useState<GlobalState[]>([]); // Initialize an empty array for processes

    const [currentID, setCurrentID] = useState<number>(0);
    const [currentStep, setCurrentStep] = useState<Screen>(Screen.Main);

    const [folderSelectionStatus, setFolderSelectionStatus] = useState({ state: TaskStates.Ready, info: "" });
    const [selectDynFolderStatus, setSelectDynFolderStatus] = useState({ state: TaskStates.Ready, info: "" });
    const [selectStaticFrameStatus, setSelectStaticFrameStatus] = useState({ state: TaskStates.Ready, info: "" });
    const [selectSideStatus, setSelectSideStatus] = useState({ state: TaskStates.NotReady, info: "" });
    const [cropDynamicFramesStatus, setCropDynamicFramesStatus] = useState({ state: TaskStates.NotReady, info: "" });
    const [gaitCycleStatus, setGaitCycleStatus] = useState({ state: TaskStates.NotReady, info: "" });
    const [pickCalibrationFramesStatus, setPickCalibrationFramesStatus] = useState({
        state: TaskStates.NotReady,
        info: "Waiting for Gait Cycle Calculation...",
    });
    const [pickCoordsStatus, setPickCoordsStatus] = useState({ state: TaskStates.NotReady, info: "" });
    const [runKinematicsStatus, setRunKinematicsStatus] = useState({ state: TaskStates.NotReady, info: "" });

    const [ID, setID] = useState(0);
    const [displayName, setDisplayName] = useState('');

    const createInitialState = (): GlobalState => {
        return {
            folderSelectionStatus,
            setFolderSelectionStatus,
            selectDynFolderStatus,
            setSelectDynFolderStatus,
            selectStaticFrameStatus,
            setSelectStaticFrameStatus,
            selectSideStatus,
            setSelectSideStatus,
            cropDynamicFramesStatus,
            setCropDynamicFramesStatus,
            gaitCycleStatus,
            setGaitCycleStatus,
            pickCalibrationFramesStatus,
            setPickCalibrationFramesStatus,
            pickCoordsStatus,
            setPickCoordsStatus,
            runKinematicsStatus,
            setRunKinematicsStatus,
            ID,
            displayName,
            setDisplayName,
        };
    };

    // Function to add a new process to the processes array
    const addProcess = (newProcess: GlobalState) => {
        setProcesses(prevProcesses => [...prevProcesses, newProcess]);
    };

    const removeProcess = (ID: number) =>{
        const copy = [...processes];
        const index = copy.findIndex((value) => {
            return value.ID == ID
        });
        copy.splice(index, 1);

        if (currentID == ID && copy.length > 0)
        {
            setCurrentID(copy[copy.length - 1].ID);
        }

        setProcesses(copy);
    }

    const createAndAddProcess = () =>
    {
        const initialState = createInitialState();
        let newID = 0
        for (const i of processes)
        {
            if (i.ID >= newID)
            {
                newID = i.ID+1;
            }
        }

        initialState.ID = newID;
        addProcess(initialState);
        setCurrentID(initialState.ID);
        return initialState;
    }

    const updateProcesses = (newState:GlobalState) =>{
        // Deep copy
        let copy = JSON.parse(JSON.stringify(processes)) as GlobalState[];
        const index = copy.findIndex((value) => value.ID == newState.ID)
        copy[index] = {...newState};
        setProcesses(copy);
    }

    const getCurrentGlobalState = () =>
    {
        const index = processes.findIndex((value) => value.ID == currentID)
        return processes[index];
    }

    return (
        <GlobalStateContext.Provider value={{
            processes,
            setProcesses,
            currentID,
            setCurrentID,
            currentStep,
            setCurrentStep,
            createAndAddProcess,
            removeProcess,
            updateProcesses,
            getCurrentGlobalState,
        }}>
            {children}
        </GlobalStateContext.Provider>
    );
};


export function useStateManager(): StateManager {
    const context = useContext(GlobalStateContext);
    if (context === undefined) {
        throw new Error('useStateManager must be used within a GlobalStateProvider');
    }
    return context;
}

export function useGlobalState(): GlobalState {
    const context = useContext(GlobalStateContext);
    if (context === undefined) {
        throw new Error('useGlobalState must be used within a GlobalStateProvider');
    }
    const index = context.processes.findIndex((value) => value.ID == context.currentID);

    return context.processes[index];
}

export const sendToServer = async (data: DataToSend): Promise<ServerResponse> => {
    try {
        const { electron } = window as any
        const response = await electron.ipcRenderer.invoke('send-data-to-server', data)
        console.log('Received server response', response)
        return response
    }
    catch (error) {
        console.error('Error sending to server:', error && error);
        return { status: 'error', message: error } as ServerResponse;
    }
}
