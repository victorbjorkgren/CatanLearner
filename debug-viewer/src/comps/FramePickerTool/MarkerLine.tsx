import React, {useEffect, useState} from "react";
import styles from './FramePickerTool.module.css';

interface MarkerLineProps {
    x: number;
    isAlive: boolean;
    isFixed: boolean;
}

export const MarkerLine: React.FC<MarkerLineProps> = ( { x, isAlive, isFixed } ) => {
    const [myPos, setMyPos] = useState<number>(0);

    useEffect(() => {
        if (!isFixed)
            setMyPos(x)
    }, [x, isFixed])

    if (!isAlive) {
        return null; // Return null instead of undefined
    }

    return (
        <div
            className={styles.markerLine}
            style={{ left: `${myPos}px` }}
        ></div>
    );
};

export default MarkerLine;
