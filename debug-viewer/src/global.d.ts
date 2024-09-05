// global.d.ts
declare module '*.module.css' {
    const classes: { [key: string]: string };
    export default classes;
}

interface ElectronAPI {
    getDebugFiles: () => [string[], strin[]];
    selectFolder: () => [string[], string, string];
    selectFile: () => string;
    newFile: () => string;
    // Add other properties or methods exposed by preload script here
}

interface Window {
    electron?: ElectronAPI;
}