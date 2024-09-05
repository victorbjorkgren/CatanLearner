import {PointSet} from "./PickCoords";

export enum BodyEnum {
    ankle,
    knee,
    hip,
    done,
}

export interface BodySet {
    ankle: any,
    knee: any,
    hip: any,
}

export interface BodyPoints {
    ankle: Point,
    knee: Point,
    hip: Point,
}

export enum CalibrationEnum {
    static="static",
    load="load",
    stance="stance",
    flex="flex",
}

export interface CalibrationSet {
    static: BodyPoints,
    load: BodyPoints,
    stance: BodyPoints,
    flex: BodyPoints,
}

export interface CalibrationFrames {
    static: string,
    load: string,
    stance: string,
    flex: string,
}

export interface CalibrationCollector {
    static: PointSet,
    load: PointSet,
    stance: PointSet,
    flex: PointSet,
}

export interface Point {
    x: number;
    y: number;
}

