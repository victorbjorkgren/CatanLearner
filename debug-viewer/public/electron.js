const { app, BrowserWindow, ipcMain, dialog} = require('electron');
const { execFile, spawn } = require('node:child_process');
const path = require('path');
const fs = require('fs');
const WebSocket = require('ws');
const find = require('find-process');
const url = require("node:url");

let reactProcess;
let reactParentProcess;

function createWindow() {
    let isDev;
    (async () => {
        isDev = (await import('electron-is-dev')).default;
    })();
    isDev = true; // DEBUG!!!
    if (!isDev)
        process.env.NODE_ENV = 'production';
    else
        process.env.NODE_ENV = 'development';

    console.log(`Running Electron in ${process.env.NODE_ENV} mode`);
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            webSecurity: false,
        },
    });
    mainWindow.webContents.openDevTools();

    let startUrl;
    console.log('dirname', __dirname);
    if (process.env.NODE_ENV === 'production') {
        startUrl = process.env.ELECTRON_START_URL || url.format({
            pathname: path.join(__dirname, '..', 'build', 'index.html'),
            protocol: 'file:',
            slashes: true,
        });
    }
    else
        startUrl = 'http://localhost:3000';

    console.log(`Loading App on ${startUrl}`)
    mainWindow.loadURL(startUrl);

}

function findFilesAndDirectories(start_path) {
    start_path = start_path.replace(/\\/g, '/'); // Replace backslashes
    let paths = fs.readdirSync(start_path); // Find all paths in folder

    let files = [];
    let directories = [];

    for (const item of paths) {
        const fullPath = path.join(start_path, item);
        if (fs.statSync(fullPath).isDirectory()) {

            // Recursively find files and directories in the subdirectory
            const [subFiles, subDirectories] = findFilesAndDirectories(fullPath);
            if (files.length > 0)// We only want directories with files in them. Not just other directories
            {
                directories.push(fullPath);
            }
            files = files.concat(subFiles);
            directories = directories.concat(subDirectories);
        } else {
            files.push(fullPath);
        }
    }

    for (let i= 0;i < directories.length; i++)
    {
        directories[i] = directories[i].replace(/\\/g, '/');
    }

    for (let i= 0;i < files.length; i++)
    {
        files[i] = files[i].replace(/\\/g, '/');
    }

    return [files, directories];
}

// Function to get sorted PNG files from a given directory
const getSortedPngFiles = (dirPath) => {
    let files = fs.readdirSync(dirPath).filter(file => file.endsWith('.png'));
    // files = files.filter(file => file.startsWith('color'));
    // files = files.filter( file => !file.endsWith('0.png'));
    files = files.map(file => path.join(dirPath, file));

    // Extract the numerical part and sort based on that
    files.sort((a, b) => {
        const numA = parseInt(path.basename(a).match(/(\d+)/)[0]);
        const numB = parseInt(path.basename(b).match(/(\d+)/)[0]);
        return numA - numB;
    });
    return files;
};

ipcMain.handle('select-folder', async () => {
    const { filePaths } = await dialog.showOpenDialog({
        properties: ['openDirectory'],
    });

    if (filePaths && filePaths.length > 0) {
        const folderPath = filePaths[0];

        // Get files from the main folder
        let files = getSortedPngFiles(folderPath);

        // If no files in the main folder, check the "color_stream" subfolder
        if (files.length === 0) {
            const colorStreamFolder = path.join(folderPath, 'color_stream');
            if (fs.existsSync(colorStreamFolder)) {
                files = getSortedPngFiles(colorStreamFolder);
            }
        }

        // Get the background file (color0.png) and remove it from the list
        const backgroundFile = files.filter(file => file.endsWith('color0.png'));
        const backgroundIndex = files.findIndex(file => file.endsWith('color0.png'));
        if (backgroundIndex !== -1) {
            files.splice(backgroundIndex, 1);
        }

        return [files, backgroundFile, folderPath];
    }
    return [[], "", ""];
});

ipcMain.handle('select-file', async () => {
    const { filePaths } = await dialog.showOpenDialog({
        properties: ['openFile'],
    });

    console.log(filePaths);

    if (filePaths && filePaths.length > 0 && filePaths[0].length > 0) {
        const file = filePaths[0];
        if (file.endsWith('.png'))
            return filePaths[0]
        else
            return "";

    }
    return [];
});

ipcMain.handle('new-file', async () => {
    const {filePath} = await dialog.showSaveDialog({
        properties: ['createDirectory', 'showOverwriteConfirmation'],
    });
    return filePath;
});

ipcMain.handle('get-debug-files', async () => {
    const { filePaths } = await dialog.showOpenDialog({
        properties: ['openDirectory']
    });

    let files;
    let directories;
    [files, directories] = findFilesAndDirectories(filePaths[0]);

    let filtered = directories.filter((element, index, array)=>{
        return array.indexOf(element) === index;
    });

    return [files, filtered]
})

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });

    // Store the process ids of the process listening on port 3000, and its parent process
    if (process.env.NODE_ENV === 'development') {
        find('port', 3000).then((list) => {
            if (list.length > 0) {
                reactProcess = list[0].pid
                reactParentProcess = list[0].ppid
            }
            else {
                console.error('No process found on port 3000');
            }
        }).catch((err) => {
            console.error('Error finding React process:', err);
        })
    }

});

app.on('before-quit', () =>{
    process.kill(reactProcess);
    process.kill(reactParentProcess);

})

app.on('window-all-closed', async () => {
    console.log('closing app...')

    if (process.platform !== 'darwin') {
        app.quit();
    }

});
