// import React from 'react';
import styles from './FramePickerTool.module.css';

export const ThumbnailComponent = ({ src, alt, width }) => {
    // Thumbnail in the line of thumbnails in
    const imageWidthStyle = { width: `${width}%` }
    return (
        <div
            className={styles.thumbnailContainer}
            style={imageWidthStyle}
        >
            <img
                src={src}
                alt={alt}
                // onMouseEnter={onMouseEnter}
                className={styles.thumbnail}
            />
        </div>
    )
}

export default ThumbnailComponent;