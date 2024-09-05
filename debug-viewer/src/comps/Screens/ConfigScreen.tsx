import TaskButton, {TaskStates} from "../Utils/TaskButton";
import {Screen, sendToServer, useStateManager} from "../../misc/Misc";
import React, {useEffect, useState} from "react";

interface ConfigScreenProps{

}

const ConfigScreen: React.FC<ConfigScreenProps> = () => {
 const sm = useStateManager();
 const [processes, setProcesses] = useState(sm.processes);
 const [exportButtonStatus, setExportButtonStatus] = useState(TaskStates.NotReady);

 function handleRemove(event: any)
 {
     const index = event.target.id;
     sm.removeProcess(index);
 }

 function handleAdd()
 {
     const newProcess = sm.createAndAddProcess();
     const newestProcessID = newProcess.ID;

     sendToServer({key: 'create', value:newProcess.ID, ID: newProcess.ID})

     sm.setCurrentID(newestProcessID);
     sm.setCurrentStep(Screen.SetupFolder);
 }

 function handleEdit(event: any)
 {
     const index = event.target.name;
     sm.setCurrentID(index);
     sm.setCurrentStep(Screen.SetupFolder);
 }

async function handleExport () {
     if (window.electron && window.electron.newFile) {
         const filePath = await window.electron.newFile();
         console.log(filePath);
         sendToServer({key: 'export-pdf', value: filePath, ID: 0})
     }
 }

 useEffect(() => {
     const temp =  [];
     for (const i of sm.processes)
     {
         temp.push(i);
     }
     if (temp.length > 0)
     {
        setExportButtonStatus(TaskStates.Ready);
     }
     else
     {
         setExportButtonStatus(TaskStates.NotReady);
     }
     setProcesses(temp);
 }, [sm.processes]);

 return (
     <div className={'config-screen'}>
         <TaskButton
             taskFunc={handleAdd}
             dispText={'Add'}
             taskStatus={{state: TaskStates.Ready, info: ""}}
         />
         <div className={'data-list'}>
             {processes.map((state, index) => (
                 <div key={state.ID} className={'list-element'}>
                     {state.ID}
                     <button name={state.ID.toString()} className={'close-button'} onClick={handleEdit}></button>
                     <button id={state.ID.toString()} className={'close-button'} onClick={handleRemove}>X</button>
                 </div>))}
         </div>
         <TaskButton
             taskFunc={handleExport}
             dispText={"Export PDF"}
             taskStatus={{state: exportButtonStatus, info: ""}}
         />
     </div>
 );
};
export default ConfigScreen;
