import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/index.css';
import './styles/global.css'
import App from './App';
import { GlobalStateProvider } from "./misc/Misc";
import reportWebVitals from './reportWebVitals';

const rootElement = document.getElementById('root');

if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(
        <React.StrictMode>
            <GlobalStateProvider>
                <App />
            </GlobalStateProvider>
        </React.StrictMode>
    );
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
