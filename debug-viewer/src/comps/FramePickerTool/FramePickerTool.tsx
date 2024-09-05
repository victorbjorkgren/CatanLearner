import React, {useEffect, useState} from 'react';
import styles from './FramePickerTool.module.css';
import ThumbnailLine from './ThumbnailLine';
import { ReferencePreview } from './ReferencePreview';
import TaskButton, {TaskStates} from "../Utils/TaskButton";
import {DataToSend, Screen, sendToServer} from "../../misc/Misc";

interface CropComponentProps {
    images: string[];
    markerCount: number;
    handleSelection: (selectedFrames: number[]) => void;
    toolTips: string[]
}

export const FramePickerTool: React.FC<CropComponentProps> = ({ images, markerCount, handleSelection, toolTips}) => {
    // Top component for the crop tool

    const [selectedFrames, setSelectedFrames] = useState<number[]>(new Array(markerCount).fill(-1));
    const [hoveredImage, setHoveredImage] = useState<string>(images[Math.floor(images.length / 2)]);
    const [currentMarkerIdx, setCurrentMarkerIdx] = useState<number>(0);
    const [message, setMessage] = useState<string>("");

    const handleReturn = () => {
        handleSelection(selectedFrames);
    }

    useEffect(()=> {
        console.log("currentMarkerIdx", currentMarkerIdx);
        if (toolTips.length === 1)
            setMessage(toolTips[0]);
        else if (currentMarkerIdx === -1)
            setMessage("Done, Please return to previous page")
        else
            setMessage(toolTips[currentMarkerIdx]);
    }, [currentMarkerIdx])


    return (
        <div className={styles.cropToolContainer}>
            <ReferencePreview image={hoveredImage} />
            <ThumbnailLine
                images={images}
                setHoveredImage={setHoveredImage}
                setSelectedFrames={setSelectedFrames}
                markerCount={markerCount}
                setCurrentMarkerIdx={setCurrentMarkerIdx}
            />
        </div>
    );
};