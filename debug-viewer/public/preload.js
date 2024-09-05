const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {

    selectFolder: async () => {
        return await ipcRenderer.invoke('select-folder');
    },

    selectFile: async () => {
        return await ipcRenderer.invoke('select-file');
    },

    newFile: async() => {
        return await ipcRenderer.invoke('new-file');
    },

    getDebugFiles: async () =>{
        return await ipcRenderer.invoke('get-debug-files');
    },

    ipcRenderer: {
        send: (channel, data) => {
            ipcRenderer.send(channel, data);
        },
        invoke: (channel, data) => {
            return ipcRenderer.invoke(channel, data);
        },
        onServerMessage: (callback) => {
            ipcRenderer.on('server-message', (event, message) => {
                callback(message);
            });
        },
        removeServerMessageListener: () => {
            ipcRenderer.removeAllListeners('server-message');
        },
    },
});
