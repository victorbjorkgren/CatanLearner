import React, {useEffect, useRef, useState} from 'react';
import styles from "./PickCoords.module.css";
import {TaskButton, TaskStates, TaskStatus} from "../Utils/TaskButton";
import {
    BodyEnum,
    BodyPoints,
    CalibrationCollector,
    CalibrationEnum,
    CalibrationFrames,
    CalibrationSet,
    Point
} from "./PickCoordTypes";

export class PointSet {
    public ankle: Point | undefined = undefined;
    public knee: Point | undefined = undefined;
    public hip: Point | undefined = undefined;

    currentlyGathering(): BodyEnum {
        if (!this.ankle) return BodyEnum.ankle;
        else if (!this.knee) return BodyEnum.knee;
        else if (!this.hip) return BodyEnum.hip;
        else return BodyEnum.done
    }

    getMessage(): string {
        if (!this.ankle) return "Click ankle";
        else if (!this.knee) return "Click knee";
        else if (!this.hip) return "Click hip";
        else return "Done"
    }

    isDone(): boolean {
        return !!(this.ankle && this.knee && this.hip)
    }

    asArray(): Array<Point> {
        let out = []
        if (this.ankle) out.push(this.ankle);
        if (this.knee) out.push(this.knee);
        if (this.hip) out.push(this.hip);
        return out
    }

    asObject(): BodyPoints {
        return {
            ankle: this.ankle ? this.ankle : {x: -1, y: -1},
            knee: this.knee ? this.knee : {x: -1, y: -1},
            hip: this.hip ? this.hip : {x: -1, y: -1},
        }
    }
}

export interface PickCoordProps {
    image: CalibrationFrames;
    handleSelection: (pointSets: CalibrationSet, isDone: boolean) => void;
}

export const PickCoords: React.FC<PickCoordProps> = ({ image, handleSelection }) => {
    // const [points, setPoints] = useState<PointSet>(new PointSet());
    const [message, setMessage] = useState<string>('Click Ankle');
    const [originalImageSize, setOriginalImageSize] = useState<{ width: number; height: number } | null>(null);
    const [imageRect, setImageRect] = useState<DOMRect | null>(null);
    const [activeFrame, setActiveFrame] = useState<CalibrationEnum>(CalibrationEnum.static);
    const [imgScale, setImgScale] = useState<Point>({x: 1, y:1})
    const [pointSets, setPointSets] = useState<CalibrationCollector>({
        static: new PointSet(),
        load: new PointSet(),
        stance: new PointSet(),
        flex: new PointSet(),
    })

    const [stanceStatus, setStanceStatus] = useState<TaskStatus>({state: TaskStates.Ready, info:""});
    const [staticStatus, setStaticStatus] = useState<TaskStatus>({state: TaskStates.Ready, info:""});
    const [flexStatus, setFlexStatus] = useState<TaskStatus>({state: TaskStates.Ready, info:""});
    const [loadStatus, setLoadStatus] = useState<TaskStatus>({state: TaskStates.Ready, info:""});
    const [isMouseOver, setIsMouseOver ] = useState(false);
    const [zoomCoordinates, setZoomCoordinates] = useState([0,0])
    const [lensCoordinates, setLensCoordinates] = useState([0, 0])
    const lensRef = useRef<HTMLDivElement>(null);
    const [zoomScale, setZoomScale] = useState(2);

    useEffect(() => {
        setImgScale({
            y: (imageRect?.height || 1) / (originalImageSize?.height || 1),
            x: (imageRect?.width || 1) / (originalImageSize?.width || 1),
        })
    }, [imageRect, originalImageSize]);

    const handleImageLoad = (event: React.SyntheticEvent<HTMLImageElement, Event>) => {
        const img = event.currentTarget;
        setOriginalImageSize({ width: img.naturalWidth, height: img.naturalHeight });
        setImageRect(event.currentTarget.getBoundingClientRect());
    };

    const handleImageClick = (event: React.MouseEvent<HTMLImageElement, MouseEvent>) => {
        if (!originalImageSize) return;

        setImageRect(event.currentTarget.getBoundingClientRect());
        if (!imageRect) return;

        const scaleX: number = originalImageSize.width / imageRect.width;
        const scaleY: number = originalImageSize.height / imageRect.height;

        if (event.clientX < imageRect.left) return
        if (event.clientY < imageRect.top) return
        if (event.clientY > imageRect.bottom) return
        if (event.clientX > imageRect.right) return

        const x: number = (event.clientX - imageRect.left) * scaleX;
        const y: number = (event.clientY - imageRect.top) * scaleY;
        const newPoint = {
            x: x,
            y: y,
        };

        const updatedPoints = pointSets[activeFrame];

        if (updatedPoints.currentlyGathering() === BodyEnum.ankle)
            updatedPoints.ankle = newPoint;
        else if (updatedPoints.currentlyGathering() === BodyEnum.knee)
            updatedPoints.knee = newPoint;
        else if (updatedPoints.currentlyGathering() === BodyEnum.hip)
            updatedPoints.hip = newPoint;

        setMessage(updatedPoints.getMessage());
        if (updatedPoints.isDone())
            markFrameDone()

        setPointSets((prevPointSets) => ({
            ...prevPointSets,
            [activeFrame]: updatedPoints,
        }));
    };

    const markFrameDone = () => {
        if (activeFrame === CalibrationEnum.static)
            setStaticStatus({ state: TaskStates.Done, info: "" });
        else if (activeFrame === CalibrationEnum.stance)
            setStanceStatus({ state: TaskStates.Done, info: "" });
        else if (activeFrame === CalibrationEnum.flex)
            setFlexStatus({ state: TaskStates.Done, info: "" });
        else if (activeFrame === CalibrationEnum.load)
            setLoadStatus({ state: TaskStates.Done, info: "" });
        else
            throw("Unknown Frame Marked Done")
    }

    const handleBack = () => {
        const isDone = (
            pointSets.static.isDone()
            && pointSets.flex.isDone()
            && pointSets.stance.isDone()
            && pointSets.load.isDone()
        )
        const returnVal = {
            static: pointSets.static.asObject(),
            load: pointSets.load.asObject(),
            stance: pointSets.stance.asObject(),
            flex: pointSets.flex.asObject(),
        }
        handleSelection(returnVal, isDone);
    }

    const handleStatic = () => {
        setActiveFrame(CalibrationEnum.static);
        setMessage(pointSets.static.getMessage());
        // setPoints(pointSets.static);
    }

    const handleLoad = () => {
        setActiveFrame(CalibrationEnum.load);
        setMessage(pointSets.load.getMessage());
        // setPoints(pointSets.load);

    }

    const handleFlex = () => {
        setActiveFrame(CalibrationEnum.flex);
        setMessage(pointSets.flex.getMessage());
        // setPoints(pointSets.flex);
    }

    const handleStance = () => {
        setActiveFrame(CalibrationEnum.stance);
        setMessage(pointSets.stance.getMessage());
        // setPoints(pointSets.stance);
    }

    const handleMouseEnter = () =>{
        setIsMouseOver(true);
        if (lensRef.current){
            lensRef.current.style.visibility = 'visible';
        }
    }

    const handleMouseLeave = () =>{
        setIsMouseOver(false);
        if (lensRef.current){
            lensRef.current.style.visibility = 'hidden';
        }
    }

    const handleMouseMove = (event : any) =>{
        if (originalImageSize && imageRect && lensRef.current) {
            const lens = lensRef.current as HTMLElement;
            const {top, left} = event.currentTarget.getBoundingClientRect();

            const x: number = (event.pageX - left - window.scrollX) * zoomScale - lens.offsetWidth/2;
            const y: number = (event.pageY - top - window.scrollY) * zoomScale - lens.offsetHeight/2;

            const lensX: number = event.pageX - left - window.scrollX;
            const lensY: number = event.pageY - top - window.scrollY;

            if (isMouseOver) {
                setZoomCoordinates([-x, y]);
                setLensCoordinates([lensX , lensY]);
            }
        }
    }

    return (
        <div className={styles.pickCoordsToolContainer}>
            Click on the image to collect points
            <div className={styles.previewContainer}>
                <img
                    className={styles.previewImage}
                    src={`file://${image[activeFrame]}`}
                    alt="Click collector"
                    onLoad={handleImageLoad}
                    onClick={handleImageClick}
                    onMouseEnter={handleMouseEnter}
                    onMouseLeave={handleMouseLeave}
                    onMouseMove={handleMouseMove}

                />
                <div className={'zoom-preview-bounds'} ref={lensRef} style={{top: lensCoordinates[1] + 'px', left: lensCoordinates[0] + 'px'}}>
                    {imageRect &&
                    <img src={`file://${image[activeFrame]}`} className={'zoom-preview'} alt={'zoom'} style={{bottom: zoomCoordinates[1] + 'px',
                        left: zoomCoordinates[0] + 'px', height: imageRect?.height * zoomScale, width: imageRect?.width * zoomScale}}>
                    </img>}
                    {lensRef.current &&
                        <div className={'zoom-preview-dot'} style={{top: lensRef.current.offsetHeight/2 + 'px', left: lensRef.current.offsetWidth/2 + 'px'}}/>
                    }
                </div>
                {pointSets[activeFrame].asArray().map((point, index) => (
                    <div
                        className={styles.previewDot}
                        key={index}
                        style={{
                            top: point.y * imgScale.y,
                            left: point.x * imgScale.x,
                        }}
                    />
                ))}
            </div>
            <div className={styles.message}>{message}</div>
            <div className={styles.switchButtons}>
                <TaskButton
                    taskFunc={handleStatic}
                    dispText="Static"
                    taskStatus={staticStatus}
                />
                <TaskButton
                    taskFunc={handleLoad}
                    dispText="Load"
                    taskStatus={loadStatus}
                />
                <TaskButton
                    taskFunc={handleStance}
                    dispText="Stance"
                    taskStatus={stanceStatus}
                />
                <TaskButton
                    taskFunc={handleFlex}
                    dispText="Flex"
                    taskStatus={flexStatus}
                />
            </div>
            <TaskButton
                taskFunc={handleBack}
                dispText="Back"
                taskStatus={{state: TaskStates.Ready, info: ""}}
            />
        </div>
    );
};