import Plot from 'react-plotly.js';
import React, {useEffect, useState} from "react";

function readTextFile(file, callback) {
    const rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status === 200) {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

const PlotlyComp = (props) => {
    const URL = props.url;
    const [plotData, setPlotData] = useState({});
    const [visibleLines, setVisibleLines] = useState([]);

    useEffect(() => {
        readTextFile(URL, (text) => {
            let data = JSON.parse(text);
            setPlotData(data);
            // Initialize visibleLines state with all lines visible
            setVisibleLines(new Array(data['data'].length).fill(true));
        });
    }, [URL]);

    const toggleLineVisibility = (index) => {
        // Toggle the visibility of the line at the specified index
        const updatedVisibility = visibleLines.map((visible, i) =>
            i === index ? !visible : visible
        );
        setVisibleLines(updatedVisibility);
    };

    return (
        <div className="PlotlyComp">
            {plotData['data'] && plotData['layout'] && (
                <>
                    <div className="checkboxes">
                        {plotData['data'].map((line, index) => (
                            <div key={index}>
                                <input
                                    type="checkbox"
                                    checked={visibleLines[index]}
                                    onChange={() => toggleLineVisibility(index)}
                                />
                                <label>{line.name || `Line ${index + 1}`}</label>
                            </div>
                        ))}
                    </div>
                    <Plot
                        data={plotData['data'].map((line, index) => ({
                            ...line,
                            visible: visibleLines[index] ? true : 'legendonly'
                        }))}
                        layout={plotData['layout']}
                        frames={plotData['frames']}
                    />
                </>
            )}
        </div>
    );
}

export default PlotlyComp;
