TODO:

    Use sync file to set timestamps
    https://github.com/AllenInstitute/visual_behavior_analysis/issues/389
    Integrate df/f from Jeds work
    Lots of pickle files have no image_set, becasue they were run with gratings.  That stimulus needs to be supported.


PyNWB:
    There is no way to construct a foreign-key relationship between an ImageSeries and an IndexSeries, with the name information available in the source data; indexseries must have int type