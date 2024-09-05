import "../Screens/Screens.module.css"
import {useEffect, useRef, useState} from "react";

interface ProgressBarProps {
    progress: number | undefined,
    visibility?: unknown
}

export const ProgressBar: React.FC<ProgressBarProps> = ({progress, visibility}) => {
    const progressBarRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (progressBarRef.current)
        {
            if (visibility) {
                progressBarRef.current.style.visibility = 'visible';
            }
            else {
                progressBarRef.current.style.visibility = 'hidden';
            }
        }
    }, [visibility]);

    return (

        <div className="progress-bar" ref={progressBarRef}>
            <progress className='gaitcycle-progress' value={progress}/>
            <img className='gaitcycle-animation' src={"./gaitcycle.gif"} alt="gif"/>
        </div>

    );
};