import React, { useEffect, useRef, useState } from "react";
import styles from './FramePickerTool.module.css';
import ThumbnailComponent from './ThumbnailComponent';
import MarkerLine from "./MarkerLine";
import {Screen} from "../../misc/Misc";
import path from "path";

// Define types for the component props
interface ThumbnailLineProps {
    images: string[];
    setHoveredImage: (image: string) => void;
    setSelectedFrames: (range: number[]) => void;
    markerCount: number;
    setCurrentMarkerIdx: React.Dispatch<React.SetStateAction<number>>;
}

const cullArrayIdx = (array: any[], N: number): number[] => {
    const length = array.length;
    const step = (length - 1) / (N - 1); // Calculate step size

    let result: number[] = [];
    for (let i = 0; i < N; i++) {
        const index = Math.round(i * step);
        if (index >= length) break;
        result.push(index);
    }

    return result;
}

export const ThumbnailLine: React.FC<ThumbnailLineProps> = ({ images, setHoveredImage, setSelectedFrames, markerCount , setCurrentMarkerIdx}) => {
    const thumbLineRef = useRef<HTMLDivElement | null>(null);
    const [thumbSize, setThumbSize] = useState<number>(100); // in %
    const [displayedImageRange, setDisplayedImageRange] = useState<number[]>([]);
    const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
    const [markerPosition, setMarkerPosition] = useState<number>(0);
    const [markerFixed, setMarkerFixed] = useState<boolean[]>(Array(markerCount).fill(false));
    const [markerIdx, setMarkerIdx] = useState<number[]>(Array(markerCount).fill(-1));

    const MIN_THUMB_PX = 50;


    const getImageNumber = (index: number): number => {
        if (index < 0 || index >= images.length)
            throw new Error("Index out of bounds");

        const filePath = images[index];
        //Replace backslash with forward slash.
        const renamedPaths = filePath.replace(/\\/g, '/')
        const fileName = renamedPaths.split('/').pop();

        console.log('fileName: ');
        console.log(fileName);

        if (!fileName)
            throw new Error("No file name in path");

        // Extract the number N using a regular expression
        const match = fileName.match(/(\d+)/);
        if (match && match[1])
            return parseInt(match[1], 10);

        throw new Error("No number found in the filename");
    }


    const updateLayout = () => {
        if (thumbLineRef.current) {
            const { width } = thumbLineRef.current.getBoundingClientRect();
            let n_images = images.length;
            const prel_pixels = width / n_images;
            if (prel_pixels <= MIN_THUMB_PX) {
                const dispArr = cullArrayIdx(images, Math.floor(width / MIN_THUMB_PX));
                setDisplayedImageRange(dispArr);
                setThumbSize(100 * (width / dispArr.length) / width);
            } else {
                const dispArr = Array.from({ length: images.length }, (_, i) => i);
                setDisplayedImageRange(dispArr);
                setThumbSize(100 * prel_pixels / width);
            }
        }
    }

    useEffect(() => {
        updateLayout(); // Run on mount to handle initial render

        const resizeObserver = new ResizeObserver(updateLayout);

        if (thumbLineRef.current)
            resizeObserver.observe(thumbLineRef.current);

        return () => {
            if (thumbLineRef.current)
                resizeObserver.unobserve(thumbLineRef.current);
        };
    }, [images]);

    const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
        if (!thumbLineRef.current) return;

        const rect = thumbLineRef.current.getBoundingClientRect();
        const newMarkerPosition = event.clientX - rect.left;
        setMarkerPosition(newMarkerPosition);

        let relativePosition = newMarkerPosition / rect.width;
        relativePosition = Math.max(0, Math.min(1, relativePosition));

        let index = Math.floor(relativePosition * images.length);
        index = Math.min(index, images.length - 1);

        setHoveredIdx(index);
        setHoveredImage(images[index]);
    }

    const idxRef = useRef(hoveredIdx);
    useEffect(() => {
        idxRef.current = hoveredIdx;
    }, [hoveredIdx]);


    const handleKeyDown = (e: KeyboardEvent) => {
        console.log('handleKeyDown', e);
        if (images.length === 0) return;
        let index = idxRef.current || 0
        if (e.key === 'ArrowLeft') {
            index--;
        } else if (e.key === 'ArrowRight') {
            index++;
        }
        index = Math.min(index, images.length - 1);
        index = Math.max(index, 0);
        setHoveredIdx(index);
        setHoveredImage(images[index]);
        console.log(index);
    };

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [images.length]);



    const handleMouseClick = () => {
        const firstUnfixedMarker = markerFixed.findIndex(fixed => !fixed);

        if (firstUnfixedMarker === -1) return; // all markers are fixed
        if (!hoveredIdx) return; // hoveredIdx not yet set

        const imageNumber = getImageNumber(hoveredIdx);

        const newMarkerFixed = [...markerFixed];
        newMarkerFixed[firstUnfixedMarker] = true;
        setMarkerFixed(newMarkerFixed);

        const newIndices = [...markerIdx];
        newIndices[firstUnfixedMarker] = imageNumber;

        setMarkerIdx(newIndices);
        setSelectedFrames(newIndices);
        setCurrentMarkerIdx(newMarkerFixed.findIndex(fixed => !fixed));
    };

    return (
        <div
            ref={thumbLineRef}
            className={styles.thumbnailLineContainer}
            onMouseMove={handleMouseMove}
            onClick={handleMouseClick}
        >
            <div className={styles.thumbnailLine}>
                {Array.from({ length: markerCount }, (_, i) => (
                    <MarkerLine
                        key={i}
                        x={markerPosition}
                        isAlive={true}
                        isFixed={markerFixed[i]}
                    />
                ))}
                {displayedImageRange.map((image_idx, index) => (
                    <ThumbnailComponent
                        key={index}
                        src={`file://${images[image_idx]}`}
                        alt={`img-${index}`}
                        width={thumbSize}
                    />
                ))}
            </div>
        </div>
    );
}

export default ThumbnailLine;