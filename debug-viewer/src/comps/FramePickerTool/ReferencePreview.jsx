import styles from './FramePickerTool.module.css';

export const ReferencePreview = ( { image }) => {
    // Window to preview frame cropping
    if ( ! image ) {
        return (
            <div className={styles.referencePreviewContainer}>Preview</div>
        )
    }
    else {
        console.log( image );
        return (
            <div className={styles.referencePreviewContainer}>
                <img
                    className={styles.referencePreviewImage}
                    src={`file://${ image }`}
                    alt={ 'Crop Frame Preview' }
                />
            </div>
        )
    }
}