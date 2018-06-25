class InputDataFields:

    image = 'image'
    original_image = 'original_image'
    key = 'key'
    source_id = 'source_id'
    filename = 'filename'
    groundtruth_text = 'groundtruth_text'
    groundtruth_keypoints = 'groundtruth_keypoints'
    lexicon = 'lexicon'


class TfExampleFields:

    image_encoded = 'image/encoded'
    image_format = 'image/format'
    filename = 'image/filename'
    channels = 'image/channels'
    colorspace = 'image/colorspace'
    height = 'image/height'
    width = 'image/width'
    source_id = 'image/source_id'
    transcript = 'image/transcript'
    lexicon = 'image/lexicon'
    keypoints = 'image/keypoints'